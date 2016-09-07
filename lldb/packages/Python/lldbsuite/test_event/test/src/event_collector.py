from __future__ import absolute_import
from __future__ import print_function

import os
import subprocess
import sys
import tempfile

# noinspection PyUnresolvedReferences
from six.moves import cPickle


def path_to_dotest_py():
    return os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        "test",
        "dotest.py")


def _make_pickled_events_filename():
    with tempfile.NamedTemporaryFile(
            prefix="lldb_test_event_pickled_event_output",
            delete=False) as temp_file:
        return temp_file.name


def _collect_events_with_command(command, events_filename):
    # Run the single test with dotest.py, outputting
    # the raw pickled events to a temp file.
    with open(os.devnull, 'w') as dev_null_file:
        subprocess.call(
            command,
            stdout=dev_null_file,
            stderr=dev_null_file)

    # Unpickle the events
    events = []
    if os.path.exists(events_filename):
        with open(events_filename, "rb") as events_file:
            while True:
                try:
                    # print("reading event")
                    event = cPickle.load(events_file)
                    # print("read event: {}".format(event))
                    if event:
                        events.append(event)
                except EOFError:
                    # This is okay.
                    break
        os.remove(events_filename)
    return events


def collect_events_whole_file(test_filename):
    events_filename = _make_pickled_events_filename()
    command = [
        sys.executable,
        path_to_dotest_py(),
        "--inferior",
        "--results-formatter=lldbsuite.test_event.formatter.pickled.RawPickledFormatter",
        "--results-file={}".format(events_filename),
        "-p",
        os.path.basename(test_filename),
        os.path.dirname(test_filename)]
    return _collect_events_with_command(command, events_filename)


def collect_events_for_directory_with_filter(test_filename, filter_desc):
    events_filename = _make_pickled_events_filename()
    command = [
        sys.executable,
        path_to_dotest_py(),
        "--inferior",
        "--results-formatter=lldbsuite.test_event.formatter.pickled.RawPickledFormatter",
        "--results-file={}".format(events_filename),
        "-f",
        filter_desc,
        os.path.dirname(test_filename)]
    return _collect_events_with_command(command, events_filename)
