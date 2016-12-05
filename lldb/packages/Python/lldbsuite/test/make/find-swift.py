#!/usr/bin/env python

# find-swift.py
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

import argparse
import os
import sys
import commands
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'plugins'))

import swift

parser = argparse.ArgumentParser()
parser.add_argument('-l', action='store_true')
parser.add_argument('-s', action='store_true')

args = parser.parse_args()

if (args.l):
    print swift.getSwiftLibraryPath()
elif (args.s):
    print swift.getSwiftSDKRoot()
else:
    print swift.getSwiftCompiler()
