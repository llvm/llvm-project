# TestSwiftDeploymentTarget.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

import lldbsuite.test.lldbinline as lldbinline

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftDeploymentTarget(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    @skipIf(macos_version=["<", "10.11"])
    @swiftTest
    def test_swift_deployment_target(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self,
                                          "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("p f", substrs=['i = 23'])

    @skipUnlessDarwin
    @skipIf(macos_version=["<", "10.11"])
    @swiftTest
    def test_swift_deployment_target_dlopen(self):
        self.build()
        # Create the target
        target = self.dbg.CreateTarget(self.getBuildArtifact("dlopen_module"))
        self.assertTrue(target, VALID_TARGET)

        (_, _, self.thread, _) = lldbutil.run_to_source_breakpoint(self,
            'break here', lldb.SBFileSpec('NewerTarget.swift'))
        self.expect("p self", substrs=['i = 23'])

