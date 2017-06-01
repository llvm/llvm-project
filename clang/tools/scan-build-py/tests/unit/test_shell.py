# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libscanbuild.shell as sut
import unittest


class ShellTest(unittest.TestCase):

    def test_encode_decode_are_same(self):
        def test(value):
            self.assertEqual(sut.encode(sut.decode(value)), value)

        test("")
        test("clang")
        test("clang this and that")

    def test_decode_encode_are_same(self):
        def test(value):
            self.assertEqual(sut.decode(sut.encode(value)), value)

        test([])
        test(['clang'])
        test(['clang', 'this', 'and', 'that'])
        test(['clang', 'this and', 'that'])
        test(['clang', "it's me", 'again'])
        test(['clang', 'some "words" are', 'quoted'])

    def test_encode(self):
        self.assertEqual(sut.encode(['clang', "it's me", 'again']),
                         'clang "it\'s me" again')
        self.assertEqual(sut.encode(['clang', "it(s me", 'again)']),
                         'clang "it(s me" "again)"')
        self.assertEqual(sut.encode(['clang', 'redirect > it']),
                         'clang "redirect > it"')
        self.assertEqual(sut.encode(['clang', '-DKEY="VALUE"']),
                         'clang -DKEY=\\"VALUE\\"')
        self.assertEqual(sut.encode(['clang', '-DKEY="value with spaces"']),
                         'clang -DKEY=\\"value with spaces\\"')
