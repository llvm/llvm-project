"""
Test that using a non-existent architecture name does not crash LLDB.
"""


import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class NoSuchArchTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Check that passing an invalid arch via the command-line fails but
        # doesn't crash
        self.expect(
            "target crete --arch nothingtoseehere %s" %
            (exe), error=True)

        # Check that passing an invalid arch via the SB API fails but doesn't
        # crash
        target = self.dbg.CreateTargetWithFileAndArch(exe, "nothingtoseehere")
        self.assertFalse(target.IsValid(), "This target should not be valid")

        # Now just create the target with the default arch and check it's fine
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "This target should now be valid")
