""" Tests that compressed debug info sections are used. """
import os
import lldb
import sys
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCompressedDebugInfo(TestBase):
  mydir = TestBase.compute_mydir(__file__)

  def setUp(self):
    TestBase.setUp(self)

  @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
  @skipUnlessPlatform(["linux"])
  def test_compressed_debug_info(self):
    """Tests that the 'frame variable' works with compressed debug info."""

    self.build()
    process = lldbutil.run_to_source_breakpoint(
        self, "main", lldb.SBFileSpec("a.c"), exe_name="compressed.out")[1]

    # The process should be stopped at a breakpoint, and the z variable should
    # be in the top frame.
    self.assertTrue(process.GetState() == lldb.eStateStopped,
                    STOPPED_DUE_TO_BREAKPOINT)
    frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
    self.assertTrue(frame.FindVariable("z").IsValid(), "z is not valid")

  @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
  @skipUnlessPlatform(["linux"])
  def test_compressed_debug_info_gnu(self):
    """Tests that the 'frame variable' works with gnu-style compressed debug info."""

    self.build()
    process = lldbutil.run_to_source_breakpoint(
        self, "main", lldb.SBFileSpec("a.c"), exe_name="compressed.gnu.out")[1]

    # The process should be stopped at a breakpoint, and the z variable should
    # be in the top frame.
    self.assertTrue(process.GetState() == lldb.eStateStopped,
                    STOPPED_DUE_TO_BREAKPOINT)
    frame = process.GetThreadAtIndex(0).GetFrameAtIndex(0)
    self.assertTrue(frame.FindVariable("z").IsValid(), "z is not valid")
