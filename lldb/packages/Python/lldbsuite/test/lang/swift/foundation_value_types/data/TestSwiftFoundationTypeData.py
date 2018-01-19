# TestSwiftFoundationValueTypes.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
import lldbsuite.test.lldbinline as lldbinline
import lldbsuite.test.decorators as decorators

# https://bugs.swift.org/browse/SR-3320
# This test fails with an assertion error with stdlib resilience enabled:
#  https://github.com/apple/swift/pull/13573
lldbinline.MakeInlineTest(
    __file__, globals(), decorators=[
        decorators.skipIf])
