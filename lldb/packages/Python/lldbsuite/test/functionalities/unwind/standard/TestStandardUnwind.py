"""
Test that we can backtrace correctly from standard functions.

This test suit is a collection of automatically generated tests from the source files in the
directory. Please DON'T add individual test cases to this file.

To add a new test case to this test suit please create a simple C/C++ application and put the
source file into the directory of the test cases. The test suit will automatically pick the
file up and generate a test case from it in run time (with name test_standard_unwind_<file_name>
after escaping some special characters).
"""

from __future__ import print_function


import unittest2
import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

test_source_dirs = ["."]


class StandardUnwindTest(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def standard_unwind_tests(self):
        # The following variables have to be defined for each architecture and OS we testing for:
        # base_function_names: List of function names where we accept that the stack unwinding is
        #                      correct if they are on the stack. It should include the bottom most
        #                      function on the stack and a list of functions where we know we can't
        #                      unwind for any reason (list of expected failure functions)
        # no_step_function_names: The list of functions where we don't want to step through
        #                         instruction by instruction for any reason. (A valid reason is if
        #                         it is impossible to step through a function instruction by
        #                         instruction because it is special for some reason.) For these
        # functions we will immediately do a step-out when we hit them.

        triple = self.dbg.GetSelectedPlatform().GetTriple()
        if re.match("arm-.*-.*-android", triple):
            base_function_names = [
                "_start",                # Base function on the stack
                "__memcpy_base",         # Function reached by a fall through from the previous function
                "__memcpy_base_aligned",
                # Function reached by a fall through from the previous function
            ]
            no_step_function_names = [
                "__sync_fetch_and_add_4",  # Calls into a special SO where we can't set a breakpoint
                "pthread_mutex_lock",
                # Uses ldrex and strex what interferes with the software single
                # stepping
                "pthread_mutex_unlock",
                # Uses ldrex and strex what interferes with the software single
                # stepping
                "pthread_once",
                # Uses ldrex and strex what interferes with the software single
                # stepping
            ]
        elif re.match("aarch64-.*-.*-android", triple):
            base_function_names = [
                "do_arm64_start",         # Base function on the stack
            ]
            no_step_function_names = [
                None,
                "__cxa_guard_acquire",
                # Uses ldxr and stxr what interferes with the software single
                # stepping
                "__cxa_guard_release",
                # Uses ldxr and stxr what interferes with the software single
                # stepping
                "pthread_mutex_lock",
                # Uses ldxr and stxr what interferes with the software single
                # stepping
                "pthread_mutex_unlock",
                # Uses ldxr and stxr what interferes with the software single
                # stepping
                "pthread_once",
                # Uses ldxr and stxr what interferes with the software single
                # stepping
            ]
        else:
            self.skipTest("No expectations for the current architecture")

        exe = os.path.join(os.getcwd(), "a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        target.BreakpointCreateByName("main")

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process is not None, "SBTarget.Launch() failed")
        self.assertEqual(
            process.GetState(),
            lldb.eStateStopped,
            "The process didn't hit main")

        index = 0
        while process.GetState() == lldb.eStateStopped:
            index += 1
            if process.GetNumThreads() > 1:
                # In case of a multi threaded inferior if one of the thread is stopped in a blocking
                # syscall and we try to step it then
                # SBThread::StepInstruction() will block forever
                self.skipTest(
                    "Multi threaded inferiors are not supported by this test")

            thread = process.GetThreadAtIndex(0)

            if self.TraceOn():
                print("INDEX: %u" % index)
                for f in thread.frames:
                    print(f)

            if thread.GetFrameAtIndex(0).GetFunctionName() is not None:
                found_main = False
                for f in thread.frames:
                    if f.GetFunctionName() in base_function_names:
                        found_main = True
                        break
                self.assertTrue(found_main,
                                "Main function isn't found on the backtrace")

            if thread.GetFrameAtIndex(
                    0).GetFunctionName() in no_step_function_names:
                thread.StepOut()
            else:
                thread.StepInstruction(False)

# Collect source files in the specified directories
test_source_files = set([])
for d in test_source_dirs:
    if os.path.isabs(d):
        dirname = d
    else:
        dirname = os.path.join(os.path.dirname(__file__), d)

    for root, _, files in os.walk(dirname):
        test_source_files = test_source_files | set(
            os.path.abspath(os.path.join(root, f)) for f in files)

# Generate test cases based on the collected source files
for f in test_source_files:
    if f.endswith(".cpp") or f.endswith(".c"):
        @add_test_categories(["dwarf"])
        @unittest2.skipIf(
            TestBase.skipLongRunningTest(),
            "Skip this long running test")
        def test_function_dwarf(self, f=f):
            if f.endswith(".cpp"):
                d = {'CXX_SOURCES': f}
            elif f.endswith(".c"):
                d = {'C_SOURCES': f}

            # If we can't compile the inferior just skip the test instead of failing it.
            # It makes the test suit more robust when testing on several different architecture
            # avoid the hassle of skipping tests manually.
            try:
                self.buildDwarf(dictionary=d)
                self.setTearDownCleanup(d)
            except:
                if self.TraceOn():
                    print(sys.exc_info()[0])
                self.skipTest("Inferior not supported")
            self.standard_unwind_tests()

        test_name = "test_unwind_" + str(f)
        for c in ".=()/\\":
            test_name = test_name.replace(c, '_')

        test_function_dwarf.__name__ = test_name
        setattr(
            StandardUnwindTest,
            test_function_dwarf.__name__,
            test_function_dwarf)
