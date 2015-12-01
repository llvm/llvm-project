#!/usr/bin/env python

"""
SWIG generation server.  Listens for connections from swig generation clients
and runs swig in the requested fashion, sending back the results.
"""

# Future imports
from __future__ import absolute_import
from __future__ import print_function

# Python modules
import argparse
import io
import logging
import os
import select
import shutil
import socket
import struct
import sys
import tempfile
import traceback

# LLDB modules
import use_lldb_suite
from lldbsuite.support import fs
from lldbsuite.support import sockutil

# package imports
from . import local
from . import remote

default_port = 8537

def process_args(args):
    # Setup the parser arguments that are accepted.
    parser = argparse.ArgumentParser(description='SWIG generation server.')

    parser.add_argument(
        "--port",
        action="store",
        default=default_port,
        help=("The local port to bind to"))

    parser.add_argument(
        "--swig-executable",
        action="store",
        default=fs.find_executable("swig"),
        dest="swig_executable")

    # Process args.
    return parser.parse_args(args)

def initialize_listening_socket(options):
    logging.debug("Creating socket...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    logging.info("Binding to ip address '', port {}".format(options.port))
    s.bind(('', options.port))

    logging.debug("Putting socket in listen mode...")
    s.listen()
    return s

def accept_once(sock, options):
    logging.debug("Waiting for connection...")
    while True:
        rlist, wlist, xlist = select.select([sock], [], [], 0.5)
        if not rlist:
            continue

        client, addr = sock.accept()
        logging.info("Received connection from {}".format(addr))
        data_size = struct.unpack("!I", sockutil.recvall(client, 4))[0]
        logging.debug("Expecting {} bytes of data from client"
                      .format(data_size))
        data = sockutil.recvall(client, data_size)
        logging.info("Received {} bytes of data from client"
                     .format(len(data)))

        pack_location = None
        try:
            tempfolder = os.path.join(tempfile.gettempdir(), "swig-bot")
            os.makedirs(tempfolder, exist_ok=True)

            pack_location = tempfile.mkdtemp(dir=tempfolder)
            logging.debug("Extracting archive to {}".format(pack_location))

            local.unpack_archive(pack_location, data)
            logging.debug("Successfully unpacked archive...")

            config_file = os.path.normpath(os.path.join(pack_location,
                                                        "config.json"))
            parsed_config = remote.parse_config(io.open(config_file))
            config = local.LocalConfig()
            config.languages = parsed_config["languages"]
            config.swig_executable = options.swig_executable
            config.src_root = pack_location
            config.target_dir = os.path.normpath(
                os.path.join(config.src_root, "output"))
            logging.info(
                "Running swig.  languages={}, swig={}, src_root={}, target={}"
                .format(config.languages, config.swig_executable,
                        config.src_root, config.target_dir))

            status = local.generate(config)
            logging.debug("Finished running swig.  Packaging up files {}"
                          .format(os.listdir(config.target_dir)))
            zip_data = io.BytesIO()
            zip_file = local.pack_archive(zip_data, config.target_dir, None)
            response_status = remote.serialize_response_status(status)
            logging.debug("Sending response status {}".format(response_status))
            logging.info("(swig output) -> swig_output.json")
            zip_file.writestr("swig_output.json", response_status)

            zip_file.close()
            response_data = zip_data.getvalue()
            logging.info("Sending {} byte response".format(len(response_data)))
            client.sendall(struct.pack("!I", len(response_data)))
            client.sendall(response_data)
        finally:
            if pack_location is not None:
                logging.debug("Removing temporary folder {}"
                              .format(pack_location))
                shutil.rmtree(pack_location)

def accept_loop(sock, options):
    while True:
        try:
            accept_once(sock, options)
        except Exception as e:
            error = traceback.format_exc()
            logging.error("An error occurred while processing the connection.")
            logging.error(error)

def run(args):
    options = process_args(args)
    print(options)
    sock = initialize_listening_socket(options)
    accept_loop(sock, options)
    return options
