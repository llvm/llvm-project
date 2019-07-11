"""
Test basics of mach core file debugging.
"""

from __future__ import print_function

import shutil
import struct

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MachCoreTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        super(MachCoreTestCase, self).setUp()
        self._initial_platform = lldb.DBG.GetSelectedPlatform()

    def tearDown(self):
        lldb.DBG.SetSelectedPlatform(self._initial_platform)
        super(MachCoreTestCase, self).tearDown()

    # This was originally marked as expected failure on Windows, but it has
    # started timing out instead, so the expectedFailure attribute no longer
    # correctly tracks it: llvm.org/pr37371
    @skipIfWindows
    def test_selected_thread(self):
        """Test that the right thread is selected after a core is loaded."""
        # Create core form YAML.
        self.yaml2obj("test.core.yaml", self.getBuildArtifact("test.core"))

        # Set debugger into synchronous mode
        self.dbg.SetAsync(False)

        # Create a target by the debugger.
        target = self.dbg.CreateTarget("")

        # Load OS plugin.
        python_os_plugin_path = os.path.join(self.getSourceDir(),
                                             'operating_system.py')
        command = "settings set target.process.python-os-plugin-path '{}'".format(
            python_os_plugin_path)
        self.dbg.HandleCommand(command)

        # Load core.
        process = target.LoadCore(self.getBuildArtifact("test.core"))
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetNumThreads(), 3)

        # Verify our OS plug-in threads showed up
        thread = process.GetThreadByID(0x111111111)
        self.assertTrue(thread.IsValid(
        ), "Make sure there is a thread 0x111111111 after we load the python OS plug-in"
                        )
        thread = process.GetThreadByID(0x222222222)
        self.assertTrue(thread.IsValid(
        ), "Make sure there is a thread 0x222222222 after we load the python OS plug-in"
                        )
        thread = process.GetThreadByID(0x333333333)
        self.assertTrue(thread.IsValid(
        ), "Make sure there is a thread 0x333333333 after we load the python OS plug-in"
                        )

        # Verify that the correct thread is selected
        thread = process.GetSelectedThread()
        self.assertEqual(thread.GetThreadID(), 0x333333333)
