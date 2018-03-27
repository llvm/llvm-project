"""
Test lldb-mi =library-loaded notifications.
"""

from __future__ import print_function


import lldbmi_testcase
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MiLibraryLoadedTestCase(lldbmi_testcase.MiTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # llvm.org/pr24452: Get lldb-mi tests working on Windows
    @skipIfFreeBSD  # llvm.org/pr22411: Failure presumably due to known thread races
    @skipIfRemote   # We do not currently support remote debugging via the MI.
    @skipIfDarwin
    def test_lldbmi_library_loaded(self):
        """Test that 'lldb-mi --interpreter' shows the =library-loaded notifications."""

        self.spawnLldbMi(args=None)

        # Load executable
        self.runCmd("-file-exec-and-symbols %s" % self.myexe)
        self.expect("\^done")

        # Test =library-loaded
        import os
        path = self.getBuildArtifact(self.myexe)
        symbols_path = os.path.join(
            path + ".dSYM",
            "Contents",
            "Resources",
            "DWARF",
            "a.out")

        def add_slashes(x): return x.replace(
            "\\",
            "\\\\").replace(
            "\"",
            "\\\"").replace(
            "\'",
            "\\\'").replace(
                "\0",
            "\\\0")
        self.expect(
            [
                "=library-loaded,id=\"%s\",target-name=\"%s\",host-name=\"%s\",symbols-loaded=\"1\",symbols-path=\"%s\",loaded_addr=\"-\",size=\"[0-9]+\"" %
                (add_slashes(path),
                 add_slashes(path),
                 add_slashes(path),
                 add_slashes(symbols_path)),
                "=library-loaded,id=\"%s\",target-name=\"%s\",host-name=\"%s\",symbols-loaded=\"0\",loaded_addr=\"-\",size=\"[0-9]+\"" %
                (add_slashes(path),
                 add_slashes(path),
                 add_slashes(path))])
