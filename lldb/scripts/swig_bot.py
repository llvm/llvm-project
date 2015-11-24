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

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Run swig-bot client or server.')

    # Arguments to control whether swig-bot runs as a client or server.
    parser.add_argument(
        "--mode",
        required=True,
        choices=["client", "server"],
        help="Run swig_bot in either client or server mode.")

    # Arguments to control logging verbosity.
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Increase logging verbosity level.")

    (options, remaining) = parser.parse_known_args(args)
    # Set logging level.
    if options.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.NOTSET
    logging.basicConfig(level=log_level)
    logging.info("logging is using level: %d", log_level)

    return (options, remaining)

if __name__ == "__main__":
    (options, remaining) = process_args(sys.argv[1:])
    try:
        if options.mode == "client":
            logging.info("Running swig_bot in client mode")
            from swig_bot_lib import client
            client.run(remaining)
        elif options.mode == "server":
            logging.info("Running swig_bot in server mode")
            from swig_bot_lib import server
            server.run(remaining)
        else:
            logging.error("Unknown mode specified.  Expected client or server.")
            sys.exit(-1)
    except KeyboardInterrupt as e:
        logging.info("Ctrl+C received.  Shutting down...")
        sys.exit(-1)
    except Exception as e:
        error = traceback.format_exc()
        logging.error("An error occurred running swig-bot.")
        logging.error(error)
