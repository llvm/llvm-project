# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libear as sut
import unittest
import os.path


class TemporaryDirectoryTest(unittest.TestCase):
    def test_creates_directory(self):
        dirname = None
        with sut.TemporaryDirectory() as tmpdir:
            self.assertTrue(os.path.isdir(tmpdir))
            dirname = tmpdir
        self.assertIsNotNone(dirname)
        self.assertFalse(os.path.exists(dirname))

    def test_removes_directory_when_exception(self):
        dirname = None
        try:
            with sut.TemporaryDirectory() as tmpdir:
                self.assertTrue(os.path.isdir(tmpdir))
                dirname = tmpdir
                raise RuntimeError('message')
        except:
            self.assertIsNotNone(dirname)
            self.assertFalse(os.path.exists(dirname))
