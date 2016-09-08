# TestPOUnwrapping.py
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
"""Test that we can correctly handle a nested generic type."""

import lldbsuite.test.lldbrepl as lldbrepl
import lldbsuite.test.decorators as decorators


class REPLBasicTestCase(lldbrepl.REPLTest):

    mydir = lldbrepl.REPLTest.compute_mydir(__file__)

    @decorators.expectedFailureAll(
        oslist=["linux"],
        bugnumber="bugs.swift.org/SR-801")
    def doTest(self):
        self.command(
            '''class Foo<T,U> {
           var t: T?
           var u: U?
           init() { t = nil; u = nil }
           init(_ x: T, _ y: U) { t = x; u = y }
        };(Foo<String,Double>(),Foo<Double,String>(3.14,"hello"))''',
            patterns=[
                r'\$R0: \(Foo<String, Double>, Foo<Double, String>\) = {',
                r'0 = {',
                r't = nil',
                r'u = nil',
                r'1 = {',
                r't = 3\.14[0-9]+', 'u = "hello"'])
