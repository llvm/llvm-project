#!/usr/bin/env python

"""
SWIG generation top-level script.  Supports both local and remote generation
of SWIG bindings for multiple languages.
"""

# Python modules
import argparse
import logging
import sys
import traceback

# LLDB modules
import use_lldb_suite

# swig_bot modules
from swig_bot_lib import client
from swig_bot_lib import server


def process_args(args):
    parser = argparse.ArgumentParser(
        description='Run swig-bot client or server.')

    # Create and populate subparser arguments for when swig_bot is
    # run in client or server mode
    subparsers = parser.add_subparsers(
        help="Pass --help to a sub-command to print detailed usage")
    client_parser = subparsers.add_parser("client",
                                          help="Run SWIG generation client")
    client.add_subparser_args(client_parser)
    client_parser.set_defaults(func=run_client)

    server_parser = subparsers.add_parser("server",
                                          help="Run SWIG generation server")
    server.add_subparser_args(server_parser)
    server_parser.set_defaults(func=run_server)

    # Arguments to control logging verbosity.
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Increase logging verbosity level.")

    options = parser.parse_args(args)
    # Set logging level.
    if options.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.NOTSET
    logging.basicConfig(level=log_level)
    logging.info("logging is using level: %d", log_level)

    return options


def run_client(options):
    logging.info("Running swig_bot in client mode")
    client.finalize_subparser_options(options)
    client.run(options)


def run_server(options):
    logging.info("Running swig_bot in server mode")
    server.finalize_subparser_options(options)
    server.run(options)

if __name__ == "__main__":
    options = process_args(sys.argv[1:])
    try:
        if options.func is None:
            logging.error(
                "Unknown mode specified.  Expected client or server.")
            sys.exit(-1)
        else:
            options.func(options)
    except KeyboardInterrupt as e:
        logging.info("Ctrl+C received.  Shutting down...")
        sys.exit(-1)
    except Exception as e:
        error = traceback.format_exc()
        logging.error("An error occurred running swig-bot.")
        logging.error(error)
