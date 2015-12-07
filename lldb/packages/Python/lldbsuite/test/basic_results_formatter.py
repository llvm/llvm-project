"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Provides basic test result output.  This is intended to be suitable for
normal LLDB test run output when no other option is specified.
"""
from __future__ import print_function

# Python system includes
import os

# Our imports
from . import test_results
import lldbsuite

class BasicResultsFormatter(test_results.ResultsFormatter):
    """Provides basic test result output."""
    @classmethod
    def arg_parser(cls):
        """@return arg parser used to parse formatter-specific options."""
        parser = super(BasicResultsFormatter, cls).arg_parser()

        parser.add_argument(
            "--assert-on-unknown-events",
            action="store_true",
            help=('cause unknown test events to generate '
                  'a python assert.  Default is to ignore.'))
        return parser

    def __init__(self, out_file, options):
        """Initializes the BasicResultsFormatter instance.
        @param out_file file-like object where formatted output is written.
        @param options_dict specifies a dictionary of options for the
        formatter.
        """
        # Initialize the parent
        super(BasicResultsFormatter, self).__init__(out_file, options)

        # self.result_event will store the most current result_event
        # by test method
        self.result_events = {}
        self.test_method_rerun_count = 0

    def handle_event(self, test_event):
        super(BasicResultsFormatter, self).handle_event(test_event)
        if test_event is None:
            return

        event_type = test_event["event"]
        if event_type is None:
            return

        if event_type == "terminate":
            self._finish_output()
        elif event_type == "test_start":
            self.track_start_time(
                test_event["test_class"],
                test_event["test_name"],
                test_event["event_time"])
        elif event_type == "test_result":
            # Build the test key.
            test_key = test_event.get("test_filename", None)
            if test_key is None:
                raise Exception(
                    "failed to find test filename for test event {}".format(
                        test_event))
            test_key += ".{}.{}".format(
                test_event.get("test_class", ""),
                test_event.get("test_name", ""))

            # Save the most recent test event for the test key.
            # This allows a second test phase to overwrite the most
            # recent result for the test key (unique per method).
            # We do final reporting at the end, so we'll report based
            # on final results.
            # We do this so that a re-run caused by, perhaps, the need
            # to run a low-load, single-worker test run can have the final
            # run's results to always be used.
            if test_key in self.result_events:
                # We are replacing the result of something that was
                # already counted by the base class.  Remove the double
                # counting by reducing by one the count for the test
                # result status.
                old_status = self.result_events[test_key]["status"]
                self.result_status_counts[old_status] -= 1

                self.test_method_rerun_count += 1
                if self.options.warn_on_multiple_results:
                    print(
                        "WARNING: test key {} already has a result: "
                        "old:{} new:{}",
                        self.result_events[test_key],
                        test_event)
            self.result_events[test_key] = test_event
        else:
            # This is an unknown event.
            if self.options.assert_on_unknown_events:
                raise Exception("unknown event type {} from {}\n".format(
                    event_type, test_event))

    def _partition_results_by_status(self, categories):
        """Partitions the captured test results by event status.

        This permits processing test results by the category ids.

        @param categories the list of categories on which to partition.
        Follows the format described in _report_category_details().

        @return a dictionary where each key is the test result status,
        and each entry is a list containing all the test result events
        that matched that test result status.  Result status IDs with
        no matching entries will have a zero-length list.
        """
        partitioned_events = {}
        for category in categories:
            result_status_id = category[0]
            matching_events = [
                [key, event] for (key, event) in self.result_events.items()
                if event.get("status", "") == result_status_id]
            partitioned_events[result_status_id] = sorted(
                matching_events,
                key=lambda x: x[1]["test_name"])
        return partitioned_events

    def _print_summary_counts(
        self, categories, result_events_by_status, extra_rows):
        """Prints summary counts for all categories.

        @param categories the list of categories on which to partition.
        Follows the format described in _report_category_details().

        @param result_events_by_status the partitioned list of test
        result events in a dictionary, with the key set to the test
        result status id and the value set to the list of test method
        results that match the status id.
        """

        # Get max length for category printed name
        category_with_max_printed_name = max(
            categories, key=lambda x: len(x[1]))
        max_category_name_length = len(category_with_max_printed_name[1])

        # If we are provided with extra rows, consider these row name lengths.
        if extra_rows is not None:
            for row in extra_rows:
                name_length = len(row[0])
                if name_length > max_category_name_length:
                    max_category_name_length = name_length

        banner_text = "Test Result Summary"
        banner_separator = "".ljust(len(banner_text), "=")

        self.out_file.write("\n{}\n{}\n{}\n".format(
            banner_separator,
            banner_text,
            banner_separator))

        # Prepend extra rows
        if extra_rows is not None:
            for row in extra_rows:
                extra_label = "{}:".format(row[0]).ljust(
                    max_category_name_length + 1)
                self.out_file.write("{} {:4}\n".format(extra_label, row[1]))

        for category in categories:
            result_status_id = category[0]
            result_label = "{}:".format(category[1]).ljust(
                max_category_name_length + 1)
            count = len(result_events_by_status[result_status_id])
            self.out_file.write("{} {:4}\n".format(
                result_label,
                count))

    @classmethod
    def _has_printable_details(cls, categories, result_events_by_status):
        """Returns whether there are any test result details that need to be printed.

        This will spin through the results and see if any result in a category
        that is printable has any results to print.

        @param categories the list of categories on which to partition.
        Follows the format described in _report_category_details().

        @param result_events_by_status the partitioned list of test
        result events in a dictionary, with the key set to the test
        result status id and the value set to the list of test method
        results that match the status id.

        @return True if there are any details (i.e. test results
        for failures, errors, unexpected successes); False otherwise.
        """
        for category in categories:
            result_status_id = category[0]
            print_matching_tests = category[2]
            if print_matching_tests:
                if len(result_events_by_status[result_status_id]) > 0:
                    # We found a printable details test result status
                    # that has details to print.
                    return True
        # We didn't find any test result category with printable
        # details.
        return False

    def _report_category_details(self, category, result_events_by_status):
        """Reports all test results matching the given category spec.

        @param category a category spec of the format [test_event_name,
        printed_category_name, print_matching_entries?]

        @param result_events_by_status the partitioned list of test
        result events in a dictionary, with the key set to the test
        result status id and the value set to the list of test method
        results that match the status id.
        """
        result_status_id = category[0]
        print_matching_tests = category[2]
        detail_label = category[3]

        if print_matching_tests:
            # Sort by test name
            for (_, event) in result_events_by_status[result_status_id]:
                test_relative_path = os.path.relpath(
                    os.path.realpath(event["test_filename"]),
                    lldbsuite.lldb_test_root)
                self.out_file.write("{}: {} ({})\n".format(
                    detail_label,
                    event["test_name"],
                    test_relative_path))

    def _finish_output_no_lock(self):
        """Writes the test result report to the output file."""
        extra_results = [
            # Total test methods processed, excluding reruns.
            ["Test Methods", len(self.result_events)],
            ["Reruns", self.test_method_rerun_count]]

        # Output each of the test result entries.
        categories = [
            # result id, printed name, print matching tests?, detail label
            [test_results.EventBuilder.STATUS_SUCCESS,
             "Success", False, None],
            [test_results.EventBuilder.STATUS_EXPECTED_FAILURE,
             "Expected Failure", False, None],
            [test_results.EventBuilder.STATUS_FAILURE,
             "Failure", True, "FAIL"],
            [test_results.EventBuilder.STATUS_ERROR, "Error", True, "ERROR"],
            [test_results.EventBuilder.STATUS_UNEXPECTED_SUCCESS,
             "Unexpected Success", True, "UNEXPECTED SUCCESS"],
            [test_results.EventBuilder.STATUS_SKIP, "Skip", False, None]]

        # Partition all the events by test result status
        result_events_by_status = self._partition_results_by_status(
            categories)

        # Print the details
        have_details = self._has_printable_details(
            categories, result_events_by_status)
        if have_details:
            self.out_file.write("\nDetails:\n")
            for category in categories:
                self._report_category_details(
                    category, result_events_by_status)

        # Print the summary
        self._print_summary_counts(
            categories, result_events_by_status, extra_results)


    def _finish_output(self):
        """Prepare and write the results report as all incoming events have
        arrived.
        """
        with self.lock:
            self._finish_output_no_lock()

    def replaces_summary(self):
        return True
