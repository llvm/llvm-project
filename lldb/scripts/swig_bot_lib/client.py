#!/usr/bin/env python

"""
SWIG generation client.  Supports both local and remote generation of SWIG
bindings for multiple languages.
"""

# Future imports
from __future__ import absolute_import
from __future__ import print_function

# Python modules
import argparse
import io
import logging
import os
import socket
import struct
import sys

# LLDB modules
import use_lldb_suite
from lldbsuite.support import fs
from lldbsuite.support import sockutil

# package imports
from . import local
from . import remote

default_ip = "127.0.0.1"
default_port = 8537

def process_args(args):
    """Returns options processed from the provided command line.

    @param args the command line to process.
    """

    # A custom action used by the --local command line option.  It can be
    # used with either 0 or 1 argument.  If used with 0 arguments, it
    # searches for a copy of swig located on the physical machine.  If
    # used with 1 argument, the argument is the path to a swig executable.
    class FindLocalSwigAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super(FindLocalSwigAction, self).__init__(
                option_strings, dest, nargs='?', **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            swig_exe = None
            if values is None:
                swig_exe = fs.find_executable('swig')
            else:
                swig_exe = values
            setattr(namespace, self.dest, os.path.normpath(swig_exe))

    # A custom action used by the --remote command line option.  It can be
    # used with either 0 or 1 arguments.  If used with 0 arguments it chooses
    # a default connection string.  If used with one argument it is a string
    # of the form `ip_address[:port]`.  If the port is unspecified, the
    # default port is used.
    class RemoteIpAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super(RemoteIpAction, self).__init__(
                option_strings, dest, nargs='?', **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            ip_port = None
            if values is None:
                ip_port = (default_ip, default_port)
            else:
                result = values.split(':')
                if len(result)==1:
                    ip_port = (result[0], default_port)
                elif len(result)==2:
                    ip_port = (result[0], int(result[1]))
                else:
                    raise ValueError("Invalid connection string")
            setattr(namespace, self.dest, ip_port)

    # Setup the parser arguments that are accepted.
    parser = argparse.ArgumentParser(
        description='Generate SWIG bindings.')

    parser.add_argument(
        "--local",
        action=FindLocalSwigAction,
        dest="swig_executable",
        help=(
            "Run the copy of swig at the specified location, or search PATH"
            "if the location is omitted"))

    parser.add_argument(
        "--remote",
        action=RemoteIpAction,
        help=(
            "Use the given connection string to connect to a remote "
            "generation service"))

    parser.add_argument(
        "--src-root",
        required=True,
        help="The root folder of the LLDB source tree.")

    parser.add_argument(
        "--target-dir",
        default=os.getcwd(),
        help=(
            "Specifies the build dir where the language binding "
            "should be placed"))

    parser.add_argument(
        "--language",
        dest="languages",
        action="append",
        help="Specifies the language to generate bindings for")

    # Process args.
    options = parser.parse_args(args)

    if options.languages is None:
        options.languages = ['python']

    if options.remote is None and options.swig_executable is None:
        logging.error("Must specify either --local or --remote")
        sys.exit(-3)

    return options

def establish_remote_connection(ip_port):
    logging.debug("Creating socket...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    logging.info("Connecting to server {} on port {}"
                 .format(ip_port[0], ip_port[1]))
    s.connect(ip_port)
    logging.info("Connection established...")
    return s

def transmit_request(connection, packed_input):
    logging.info("Sending {} bytes of compressed data."
                 .format(len(packed_input)))
    connection.sendall(struct.pack("!I", len(packed_input)))
    connection.sendall(packed_input)
    logging.info("Awaiting response.")
    response_len = struct.unpack("!I", sockutil.recvall(connection, 4))[0]
    logging.debug("Expecting {} byte response".format(response_len))
    response = sockutil.recvall(connection, response_len)
    return response

def handle_response(options, connection, response):
    logging.debug("Received {} byte response.".format(len(response)))
    logging.debug("Creating output directory {}"
                    .format(options.target_dir))
    os.makedirs(options.target_dir, exist_ok=True)

    logging.info("Unpacking response archive into {}"
                    .format(options.target_dir))
    local.unpack_archive(options.target_dir, response)
    response_file_path = os.path.normpath(
        os.path.join(options.target_dir, "swig_output.json"))
    if not os.path.isfile(response_file_path):
        logging.error("Response file '{}' does not exist."
                        .format(response_file_path))
        return
    try:
        response = remote.deserialize_response_status(
            io.open(response_file_path))
        if response[0] != 0:
            logging.error("An error occurred during generation.  Status={}"
                            .format(response[0]))
            logging.error(response[1])
        else:
            logging.info("SWIG generation successful.")
            if len(response[1]) > 0:
                logging.info(response[1])
    finally:
        os.unlink(response_file_path)

def run(args):
    options = process_args(args)

    if options.remote is None:
        logging.info("swig bot client using local swig installation at '{}'"
                     .format(options.swig_executable))
        if not os.path.isfile(options.swig_executable):
            logging.error("Swig executable '{}' does not exist."
                          .format(options.swig_executable))
        config = local.LocalConfig()
        config.languages = options.languages
        config.src_root = options.src_root
        config.target_dir = options.target_dir
        config.swig_executable = options.swig_executable
        local.generate(config)
    else:
        logging.info("swig bot client using remote generation with server '{}'"
                     .format(options.remote))
        connection = None
        try:
            config = remote.generate_config(options.languages)
            logging.debug("Generated config json {}".format(config))
            inputs = [("include/lldb", ".h"),
                      ("include/lldb/API", ".h"),
                      ("scripts", ".swig"),
                      ("scripts/Python", ".swig"),
                      ("scripts/interface", ".i")]
            zip_data = io.BytesIO()
            packed_input = local.pack_archive(zip_data, options.src_root, inputs)
            logging.info("(null) -> config.json")
            packed_input.writestr("config.json", config)
            packed_input.close()
            connection = establish_remote_connection(options.remote)
            response = transmit_request(connection, zip_data.getvalue())
            handle_response(options, connection, response)
        finally:
            if connection is not None:
                connection.close()