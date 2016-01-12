# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libscanbuild.command as sut
from . import fixtures
import unittest


class ParseTest(unittest.TestCase):

    def test_action(self):
        def test(expected, cmd):
            opts = sut.classify_parameters(cmd)
            self.assertEqual(expected, opts['action'])

        Link = sut.Action.Link
        test(Link, ['clang', 'source.c'])

        Compile = sut.Action.Compile
        test(Compile, ['clang', '-c', 'source.c'])
        test(Compile, ['clang', '-c', 'source.c', '-MF', 'source.d'])

        Preprocess = sut.Action.Ignored
        test(Preprocess, ['clang', '-E', 'source.c'])
        test(Preprocess, ['clang', '-c', '-E', 'source.c'])
        test(Preprocess, ['clang', '-c', '-M', 'source.c'])
        test(Preprocess, ['clang', '-c', '-MM', 'source.c'])

    def test_optimalizations(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('compile_options', [])

        self.assertEqual(['-O'],  test(['clang', '-c', 'source.c', '-O']))
        self.assertEqual(['-O1'], test(['clang', '-c', 'source.c', '-O1']))
        self.assertEqual(['-Os'], test(['clang', '-c', 'source.c', '-Os']))
        self.assertEqual(['-O2'], test(['clang', '-c', 'source.c', '-O2']))
        self.assertEqual(['-O3'], test(['clang', '-c', 'source.c', '-O3']))

    def test_language(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('language')

        self.assertEqual(None, test(['clang', '-c', 'source.c']))
        self.assertEqual('c', test(['clang', '-c', 'source.c', '-x', 'c']))
        self.assertEqual('cpp', test(['clang', '-c', 'source.c', '-x', 'cpp']))

    def test_output(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('output')

        self.assertEqual(None, test(['clang', '-c', 'source.c']))
        self.assertEqual('source.o',
                         test(['clang', '-c', '-o', 'source.o', 'source.c']))

    def test_arch(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('archs_seen', [])

        eq = self.assertEqual

        eq([], test(['clang', '-c', 'source.c']))
        eq(['mips'],
           test(['clang', '-c', 'source.c', '-arch', 'mips']))
        eq(['mips', 'i386'],
           test(['clang', '-c', 'source.c', '-arch', 'mips', '-arch', 'i386']))

    def test_input_file(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('files', [])

        eq = self.assertEqual

        eq(['src.c'], test(['clang', 'src.c']))
        eq(['src.c'], test(['clang', '-c', 'src.c']))
        eq(['s1.c', 's2.c'], test(['clang', '-c', 's1.c', 's2.c']))

    def test_include(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('compile_options', [])

        eq = self.assertEqual

        eq([], test(['clang', '-c', 'src.c']))
        eq(['-include', '/usr/local/include'],
           test(['clang', '-c', 'src.c', '-include', '/usr/local/include']))
        eq(['-I.'],
           test(['clang', '-c', 'src.c', '-I.']))
        eq(['-I', '.'],
           test(['clang', '-c', 'src.c', '-I', '.']))
        eq(['-I/usr/local/include'],
           test(['clang', '-c', 'src.c', '-I/usr/local/include']))
        eq(['-I', '/usr/local/include'],
           test(['clang', '-c', 'src.c', '-I', '/usr/local/include']))
        eq(['-I/opt', '-I', '/opt/otp/include'],
           test(['clang', '-c', 'src.c', '-I/opt', '-I', '/opt/otp/include']))
        eq(['-isystem', '/path'],
           test(['clang', '-c', 'src.c', '-isystem', '/path']))
        eq(['-isystem=/path'],
           test(['clang', '-c', 'src.c', '-isystem=/path']))

    def test_define(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('compile_options', [])

        eq = self.assertEqual

        eq([], test(['clang', '-c', 'src.c']))
        eq(['-DNDEBUG'],
           test(['clang', '-c', 'src.c', '-DNDEBUG']))
        eq(['-UNDEBUG'],
           test(['clang', '-c', 'src.c', '-UNDEBUG']))
        eq(['-Dvar1=val1', '-Dvar2=val2'],
           test(['clang', '-c', 'src.c', '-Dvar1=val1', '-Dvar2=val2']))
        eq(['-Dvar="val ues"'],
           test(['clang', '-c', 'src.c', '-Dvar="val ues"']))

    def test_ignored_flags(self):
        def test(flags):
            cmd = ['clang', 'src.o']
            opts = sut.classify_parameters(cmd + flags)
            self.assertEqual(['src.o'], opts.get('compile_options'))

        test([])
        test(['-lrt', '-L/opt/company/lib'])
        test(['-static'])
        test(['-Wnoexcept', '-Wall'])
        test(['-mtune=i386', '-mcpu=i386'])

    def test_compile_only_flags(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('compile_options', [])

        eq = self.assertEqual

        eq(['-std=C99'],
           test(['clang', '-c', 'src.c', '-std=C99']))
        eq(['-nostdinc'],
           test(['clang', '-c', 'src.c', '-nostdinc']))
        eq(['-isystem', '/image/debian'],
           test(['clang', '-c', 'src.c', '-isystem', '/image/debian']))
        eq(['-iprefix', '/usr/local'],
           test(['clang', '-c', 'src.c', '-iprefix', '/usr/local']))
        eq(['-iquote=me'],
           test(['clang', '-c', 'src.c', '-iquote=me']))
        eq(['-iquote', 'me'],
           test(['clang', '-c', 'src.c', '-iquote', 'me']))

    def test_compile_and_link_flags(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('compile_options', [])

        eq = self.assertEqual

        eq(['-fsinged-char'],
           test(['clang', '-c', 'src.c', '-fsinged-char']))
        eq(['-fPIC'],
           test(['clang', '-c', 'src.c', '-fPIC']))
        eq(['-stdlib=libc++'],
           test(['clang', '-c', 'src.c', '-stdlib=libc++']))
        eq(['--sysroot', '/'],
           test(['clang', '-c', 'src.c', '--sysroot', '/']))
        eq(['-isysroot', '/'],
           test(['clang', '-c', 'src.c', '-isysroot', '/']))
        eq([],
           test(['clang', '-c', 'src.c', '-fsyntax-only']))
        eq([],
           test(['clang', '-c', 'src.c', '-sectorder', 'a', 'b', 'c']))

    def test_detect_cxx_from_compiler_name(self):
        def test(cmd):
            opts = sut.classify_parameters(cmd)
            return opts.get('c++')

        eq = self.assertEqual

        eq(False, test(['cc', '-c', 'src.c']))
        eq(True, test(['c++', '-c', 'src.c']))
        eq(False, test(['clang', '-c', 'src.c']))
        eq(True, test(['clang++', '-c', 'src.c']))
        eq(False, test(['gcc', '-c', 'src.c']))
        eq(True, test(['g++', '-c', 'src.c']))
