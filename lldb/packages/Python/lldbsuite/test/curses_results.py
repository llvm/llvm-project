#!/usr/bin/env python

"""
                     The LLVM Compiler Infrastructure

 This file is distributed under the University of Illinois Open Source
 License. See LICENSE.TXT for details.

Configuration options for lldbtest.py set by dotest.py during initialization
"""

from __future__ import print_function
from __future__ import absolute_import

# System modules
import curses
import datetime
import math
import sys
import time

# Third-party modules

# LLDB modules
from . import lldbcurses
from . import result_formatter
from .result_formatter import EventBuilder


class Curses(result_formatter.ResultsFormatter):
    """Receives live results from tests that are running and reports them to the terminal in a curses GUI"""

    def __init__(self, out_file, options):
        # Initialize the parent
        super(Curses, self).__init__(out_file, options)
        self.using_terminal = True
        self.have_curses = True
        self.initialize_event = None
        self.jobs = [None] * 64
        self.job_tests = [None] * 64
        self.results = list()
        try:
            self.main_window = lldbcurses.intialize_curses()
            self.main_window.add_key_action('\t', self.main_window.select_next_first_responder, "Switch between views that can respond to keyboard input")
            self.main_window.refresh()
            self.job_panel = None
            self.results_panel = None
            self.status_panel = None
            self.info_panel = None
            self.hide_status_list = list()
            self.start_time = time.time()
        except:
            self.have_curses = False
            lldbcurses.terminate_curses()
            self.using_terminal = False
            print("Unexpected error:", sys.exc_info()[0])
            raise

        self.line_dict = dict()
        # self.events_file = open("/tmp/events.txt", "w")
        # self.formatters = list()
        # if tee_results_formatter:
        #     self.formatters.append(tee_results_formatter)

    def status_to_short_str(self, status):
        if status == EventBuilder.STATUS_SUCCESS:
            return '.'
        elif status == EventBuilder.STATUS_FAILURE:
            return 'F'
        elif status == EventBuilder.STATUS_UNEXPECTED_SUCCESS:
            return '?'
        elif status == EventBuilder.STATUS_EXPECTED_FAILURE:
            return 'X'
        elif status == EventBuilder.STATUS_SKIP:
            return 'S'
        elif status == EventBuilder.STATUS_ERROR:
            return 'E'
        else:
            return status

    def show_info_panel(self):
        selected_idx = self.results_panel.get_selected_idx()
        if selected_idx >= 0 and selected_idx < len(self.results):
            if self.info_panel is None:
                info_frame = self.results_panel.get_contained_rect(top_inset=10, left_inset=10, right_inset=10, height=30)
                self.info_panel = lldbcurses.BoxedPanel(info_frame, "Result Details")
                # Add a key action for any key that will hide this panel when any key is pressed
                self.info_panel.add_key_action(-1, self.hide_info_panel, 'Hide the info panel')
                self.info_panel.top()
            else:
                self.info_panel.show()

            self.main_window.push_first_responder(self.info_panel)
            test_start = self.results[selected_idx][0]
            test_result = self.results[selected_idx][1]
            self.info_panel.set_line(0, "File: %s" % (test_start['test_filename']))
            self.info_panel.set_line(1, "Test: %s.%s" % (test_start['test_class'], test_start['test_name']))
            self.info_panel.set_line(2, "Time: %s" % (test_result['elapsed_time']))
            self.info_panel.set_line(3, "Status: %s" % (test_result['status']))

    def hide_info_panel(self):
        self.main_window.pop_first_responder(self.info_panel)
        self.info_panel.hide()
        self.main_window.refresh()

    def toggle_status(self, status):
        if status:
            # Toggle showing and hiding results whose status matches "status" in "Results" window
            if status in self.hide_status_list:
                self.hide_status_list.remove(status)
            else:
                self.hide_status_list.append(status)
            self.update_results()

    def update_results(self, update=True):
        '''Called after a category of test have been show/hidden to update the results list with
           what the user desires to see.'''
        self.results_panel.clear(update=False)
        for result in self.results:
            test_result = result[1]
            status = test_result['status']
            if status in self.hide_status_list:
                continue
            name = test_result['test_class'] + '.' + test_result['test_name']
            self.results_panel.append_line('%s (%6.2f sec) %s' % (self.status_to_short_str(status), test_result['elapsed_time'], name))
        if update:
            self.main_window.refresh()

    def handle_event(self, test_event):
        with self.lock:
            super(Curses, self).handle_event(test_event)
            # for formatter in self.formatters:
            #     formatter.process_event(test_event)
            if self.have_curses:
                worker_index = -1
                if 'worker_index' in test_event:
                    worker_index = test_event['worker_index']
                if 'event' in test_event:
                    check_for_one_key = True
                    #print(str(test_event), file=self.events_file)
                    event = test_event['event']
                    if self.status_panel:
                        self.status_panel.update_status('time', str(datetime.timedelta(seconds=math.floor(time.time() - self.start_time))))
                    if event == 'test_start':
                        name = test_event['test_class'] + '.' + test_event['test_name']
                        self.job_tests[worker_index] = test_event
                        if 'pid' in test_event:
                            line = 'pid: %5d ' % (test_event['pid']) + name
                        else:
                            line = name
                        self.job_panel.set_line(worker_index, line)
                        self.main_window.refresh()
                    elif event == 'test_result':
                        status = test_event['status']
                        self.status_panel.increment_status(status)
                        if 'pid' in test_event:
                            line = 'pid: %5d ' % (test_event['pid'])
                        else:
                            line = ''
                        self.job_panel.set_line(worker_index, line)
                        name = test_event['test_class'] + '.' + test_event['test_name']
                        elapsed_time = test_event['event_time'] - self.job_tests[worker_index]['event_time']
                        if not status in self.hide_status_list:
                            self.results_panel.append_line('%s (%6.2f sec) %s' % (self.status_to_short_str(status), elapsed_time, name))
                        self.main_window.refresh()
                        # Append the result pairs
                        test_event['elapsed_time'] = elapsed_time
                        self.results.append([self.job_tests[worker_index], test_event])
                        self.job_tests[worker_index] = ''
                    elif event == 'job_begin':
                        self.jobs[worker_index] = test_event
                        if 'pid' in test_event:
                            line = 'pid: %5d ' % (test_event['pid'])
                        else:
                            line = ''
                        self.job_panel.set_line(worker_index, line)
                    elif event == 'job_end':
                        self.jobs[worker_index] = ''
                        self.job_panel.set_line(worker_index, '')
                    elif event == 'initialize':
                        self.initialize_event = test_event
                        num_jobs = test_event['worker_count']
                        job_frame = self.main_window.get_contained_rect(height=num_jobs+2)
                        results_frame = self.main_window.get_contained_rect(top_inset=num_jobs+2, bottom_inset=1)
                        status_frame = self.main_window.get_contained_rect(height=1, top_inset=self.main_window.get_size().h-1)
                        self.job_panel = lldbcurses.BoxedPanel(frame=job_frame, title="Jobs")
                        self.results_panel = lldbcurses.BoxedPanel(frame=results_frame, title="Results")

                        self.results_panel.add_key_action(curses.KEY_UP,    self.results_panel.select_prev      , "Select the previous list entry")
                        self.results_panel.add_key_action(curses.KEY_DOWN,  self.results_panel.select_next      , "Select the next list entry")
                        self.results_panel.add_key_action(curses.KEY_HOME,  self.results_panel.scroll_begin     , "Scroll to the start of the list")
                        self.results_panel.add_key_action(curses.KEY_END,   self.results_panel.scroll_end       , "Scroll to the end of the list")
                        self.results_panel.add_key_action(curses.KEY_ENTER, self.show_info_panel                , "Display info for the selected result item")
                        self.results_panel.add_key_action('.', lambda : self.toggle_status(EventBuilder.STATUS_SUCCESS)           , "Toggle showing/hiding tests whose status is 'success'")
                        self.results_panel.add_key_action('e', lambda : self.toggle_status(EventBuilder.STATUS_ERROR)             , "Toggle showing/hiding tests whose status is 'error'")
                        self.results_panel.add_key_action('f', lambda : self.toggle_status(EventBuilder.STATUS_FAILURE)           , "Toggle showing/hiding tests whose status is 'failure'")
                        self.results_panel.add_key_action('s', lambda : self.toggle_status(EventBuilder.STATUS_SKIP)              , "Toggle showing/hiding tests whose status is 'skip'")
                        self.results_panel.add_key_action('x', lambda : self.toggle_status(EventBuilder.STATUS_EXPECTED_FAILURE)  , "Toggle showing/hiding tests whose status is 'expected_failure'")
                        self.results_panel.add_key_action('?', lambda : self.toggle_status(EventBuilder.STATUS_UNEXPECTED_SUCCESS), "Toggle showing/hiding tests whose status is 'unexpected_success'")
                        self.status_panel = lldbcurses.StatusPanel(frame=status_frame)

                        self.main_window.add_child(self.job_panel)
                        self.main_window.add_child(self.results_panel)
                        self.main_window.add_child(self.status_panel)
                        self.main_window.set_first_responder(self.results_panel)

                        self.status_panel.add_status_item(name="time", title="Elapsed", format="%s", width=20, value="0:00:00", update=False)
                        self.status_panel.add_status_item(name=EventBuilder.STATUS_SUCCESS, title="Success", format="%u", width=20, value=0, update=False)
                        self.status_panel.add_status_item(name=EventBuilder.STATUS_FAILURE, title="Failure", format="%u", width=20, value=0, update=False)
                        self.status_panel.add_status_item(name=EventBuilder.STATUS_ERROR, title="Error", format="%u", width=20, value=0, update=False)
                        self.status_panel.add_status_item(name=EventBuilder.STATUS_SKIP, title="Skipped", format="%u", width=20, value=0, update=True)
                        self.status_panel.add_status_item(name=EventBuilder.STATUS_EXPECTED_FAILURE, title="Expected Failure", format="%u", width=30, value=0, update=False)
                        self.status_panel.add_status_item(name=EventBuilder.STATUS_UNEXPECTED_SUCCESS, title="Unexpected Success", format="%u", width=30, value=0, update=False)
                        self.main_window.refresh()
                    elif event == 'terminate':
                        #self.main_window.key_event_loop()
                        lldbcurses.terminate_curses()
                        check_for_one_key = False
                        self.using_terminal = False
                        # Check for 1 keypress with no delay

                    # Check for 1 keypress with no delay
                    if check_for_one_key:
                        self.main_window.key_event_loop(0, 1)
