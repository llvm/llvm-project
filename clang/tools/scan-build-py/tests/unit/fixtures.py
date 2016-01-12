# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import contextlib
import tempfile
import shutil
import unittest


class Spy(object):
    def __init__(self):
        self.arg = None
        self.success = 0

    def call(self, params):
        self.arg = params
        return self.success


@contextlib.contextmanager
def TempDir():
    name = tempfile.mkdtemp(prefix='scan-build-test-')
    try:
        yield name
    finally:
        shutil.rmtree(name)


class TestCase(unittest.TestCase):
    def assertIn(self, element, collection):
        found = False
        for it in collection:
            if element == it:
                found = True

        self.assertTrue(found, '{0} does not have {1}'.format(collection,
                                                              element))
