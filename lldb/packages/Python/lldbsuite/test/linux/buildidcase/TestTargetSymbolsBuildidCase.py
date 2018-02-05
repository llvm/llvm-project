""" Testing separate debug info loading by its .build-id. """
import os
import time
import lldb
import sys
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTargetSymbolsBuildidCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test  # Prevent the genaration of the dwarf version of this test
    @skipUnlessPlatform(['linux'])
    def test_target_symbols_buildid_case(self):
        self.build(clean=True)
        exe = self.getBuildArtifact("stripped.out")

        lldbutil.run_to_name_breakpoint(self, "main", exe_name = exe)
