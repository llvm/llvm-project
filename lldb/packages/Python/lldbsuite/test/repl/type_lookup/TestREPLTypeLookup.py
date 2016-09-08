# TestREPLTypeLookup.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See http://swift.org/LICENSE.txt for license information
# See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""Test that type lookup chooses the right language"""

import lldbsuite.test.lldbrepl as lldbrepl
import lldbsuite.test.decorators as decorators


class REPLTypeLookupTestCase(lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    @decorators.swiftTest
    @decorators.skipUnlessDarwin
    @decorators.no_debug_info_test
    def doTest(self):
        self.command(
            ':type lookup NSArchiver',
            patterns=['@interface NSArchiver'])  # no Swift info, ObjC
        self.command('import Foundation')
        self.command(
            ':type lookup NSArchiver',
            patterns=['class NSArchiver'])  # Swift info, no ObjC
