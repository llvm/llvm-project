#!/usr/bin/env python
"""Inferior program used by process control tests."""

from __future__ import print_function

import argparse
import datetime
import signal
import subprocess
import sys
import time


def parse_args(command_line):
    """Parses the command line arguments given to it.

    @param command_line a list of command line arguments to be parsed.

    @return the argparse options dictionary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore-signal",
        "-i",
        dest="ignore_signals",
        metavar="SIGNUM",
        action="append",
        type=int,
        default=[],
        help="ignore the given signal number (if possible)")
    parser.add_argument(
        "--launch-child-share-handles",
        action="store_true",
        help=("launch a child inferior.py that shares stdout/stderr/stdio and "
              "never returns"))
    parser.add_argument(
        "--never-return",
        action="store_true",
        help="run in an infinite loop, never return")
    parser.add_argument(
        "--return-code",
        "-r",
        type=int,
        default=0,
        help="specify the return code for the inferior upon exit")
    parser.add_argument(
        "--sleep",
        "-s",
        metavar="SECONDS",
        dest="sleep_seconds",
        type=float,
        help="sleep for SECONDS seconds before returning")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="log verbose operation details to stdout")
    return parser.parse_args(command_line)


def handle_ignore_signals(options, signals):
    """Ignores any signals provided to it.

    @param options the command line options parsed by the program.
    General used to check flags for things like verbosity.

    @param signals the list of signals to ignore.  Can be None or zero-length.
    Entries should be type int.
    """
    if signals is None:
        return

    for signum in signals:
        if options.verbose:
            print("disabling signum {}".format(signum))
        signal.signal(signum, signal.SIG_IGN)


def handle_sleep(options, sleep_seconds):
    """Sleeps the number of seconds specified, restarting as needed.

    @param options the command line options parsed by the program.
    General used to check flags for things like verbosity.

    @param sleep_seconds the number of seconds to sleep.  If None
    or <= 0, no sleeping will occur.
    """
    if sleep_seconds is None:
        return

    if sleep_seconds <= 0:
        return

    end_time = datetime.datetime.now() + datetime.timedelta(0, sleep_seconds)
    if options.verbose:
        print("sleep end time: {}".format(end_time))

    # Do sleep in a loop: signals can interrupt.
    while datetime.datetime.now() < end_time:
        # We'll wrap this in a try/catch so we don't encounter
        # a race if a signal (ignored) knocks us out of this
        # loop and causes us to return.
        try:
            sleep_interval = end_time - datetime.datetime.now()
            sleep_seconds = sleep_interval.total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        except:  # pylint: disable=bare-except
            pass


def handle_launch_children(options):
    if options.launch_child_share_handles:
        # Launch the child, share our file handles.
        # We won't bother reaping it since it will likely outlive us.
        subprocess.Popen([sys.executable, __file__, "--never-return"])


def handle_never_return(options):
    if not options.never_return:
        return

    # Loop forever.
    while True:
        try:
            time.sleep(10)
        except:  # pylint: disable=bare-except
            # Ignore
            pass


def main(command_line):
    """Drives the main operation of the inferior test program.

    @param command_line the command line options to process.

    @return the exit value (program return code) for the process.
    """
    options = parse_args(command_line)
    handle_ignore_signals(options, options.ignore_signals)
    handle_launch_children(options)
    handle_sleep(options, options.sleep_seconds)
    handle_never_return(options)

    return options.return_code

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
