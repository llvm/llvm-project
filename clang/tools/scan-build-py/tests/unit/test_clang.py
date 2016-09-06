# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libear
import libscanbuild.clang as sut
import unittest
import os.path


class GetClangArgumentsTest(unittest.TestCase):
    def test_get_clang_arguments(self):
        with libear.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.c')
            with open(filename, 'w') as handle:
                handle.write('')

            result = sut.get_arguments(
                ['clang', '-c', filename, '-DNDEBUG', '-Dvar="this is it"'],
                tmpdir)

            self.assertTrue('NDEBUG' in result)
            self.assertTrue('var="this is it"' in result)

    def test_get_clang_arguments_fails(self):
        self.assertRaises(
            Exception, sut.get_arguments,
            ['clang', '-###', '-fsyntax-only', '-x', 'c', 'notexist.c'], '.')


class GetCheckersTest(unittest.TestCase):
    def test_get_checkers(self):
        # this test is only to see is not crashing
        result = sut.get_checkers('clang', [])
        self.assertTrue(len(result))

    def test_get_active_checkers(self):
        # this test is only to see is not crashing
        result = sut.get_active_checkers('clang', [])
        self.assertTrue(len(result))
