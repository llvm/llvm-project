"""
Test that loading of dependents works correctly for all the potential
combinations.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

@skipIfWindows # Windows deals differently with shared libs.
class TargetDependentsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.build()

    @skipIf(bugnumber='rdar://44831253', oslist=['linux'])
    def has_exactly_one_image(self, matching, msg=""):
        self.expect(
            "image list",
            "image list should contain at least one image",
            substrs=['[  0]'])
        should_match = not matching
        self.expect(
            "image list", msg, matching=should_match, substrs=['[  1]'])

    @expectedFailureAll(oslist=["linux"]) #linux does not support loading dependent files
    def test_dependents_implicit_default_exe(self):
        """Test default behavior"""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("target create  " + exe, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(False)

    @expectedFailureAll(oslist=["linux"]) #linux does not support loading dependent files
    def test_dependents_explicit_default_exe(self):
        """Test default behavior"""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("target create -ddefault " + exe, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(False)

    @skipIf(bugnumber='rdar://44831253', oslist=['linux'])
    def test_dependents_explicit_true_exe(self):
        """Test default behavior"""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("target create -dtrue " + exe, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(True)

    @expectedFailureAll(oslist=["linux"]) #linux does not support loading dependent files
    def test_dependents_explicit_false_exe(self):
        """Test default behavior"""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("target create -dfalse " + exe, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(False)

    @skipIf(bugnumber='rdar://44831253', oslist=['linux'])
    def test_dependents_implicit_false_exe(self):
        """Test default behavior"""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("target create  -d " + exe, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(True)

    @skipIf(bugnumber='rdar://44831253', oslist=['linux'])
    def test_dependents_implicit_default_lib(self):
        ctx = self.platformContext
        dylibName = ctx.shlib_prefix + 'load_a.' + ctx.shlib_extension
        lib = self.getBuildArtifact(dylibName)
        self.runCmd("target create " + lib, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(True)

    @skipIf(bugnumber='rdar://44831253', oslist=['linux'])
    def test_dependents_explicit_default_lib(self):
        ctx = self.platformContext
        dylibName = ctx.shlib_prefix + 'load_a.' + ctx.shlib_extension
        lib = self.getBuildArtifact(dylibName)
        self.runCmd("target create -ddefault " + lib, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(True)

    @skipIf(bugnumber='rdar://44831253', oslist=['linux'])
    def test_dependents_explicit_true_lib(self):
        ctx = self.platformContext
        dylibName = ctx.shlib_prefix + 'load_a.' + ctx.shlib_extension
        lib = self.getBuildArtifact(dylibName)
        self.runCmd("target create -dtrue " + lib, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(True)

    @expectedFailureAll(oslist=["linux"]) #linux does not support loading dependent files
    def test_dependents_explicit_false_lib(self):
        ctx = self.platformContext
        dylibName = ctx.shlib_prefix + 'load_a.' + ctx.shlib_extension
        lib = self.getBuildArtifact(dylibName)
        self.runCmd("target create -dfalse " + lib, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(False)

    @skipIf(bugnumber='rdar://44831253', oslist=['linux'])
    def test_dependents_implicit_false_lib(self):
        ctx = self.platformContext
        dylibName = ctx.shlib_prefix + 'load_a.' + ctx.shlib_extension
        lib = self.getBuildArtifact(dylibName)
        self.runCmd("target create -d " + lib, CURRENT_EXECUTABLE_SET)
        self.has_exactly_one_image(True)
