# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libear
import libscanbuild.runner as sut
import unittest
import re
import os
import os.path


class FilteringFlagsTest(unittest.TestCase):

    def test_language_captured(self):
        def test(flags):
            cmd = ['clang', '-c', 'source.c'] + flags
            opts = sut.classify_parameters(cmd)
            return opts['language']

        self.assertEqual(None, test([]))
        self.assertEqual('c', test(['-x', 'c']))
        self.assertEqual('cpp', test(['-x', 'cpp']))

    def test_arch(self):
        def test(flags):
            cmd = ['clang', '-c', 'source.c'] + flags
            opts = sut.classify_parameters(cmd)
            return opts['arch_list']

        self.assertEqual([], test([]))
        self.assertEqual(['mips'], test(['-arch', 'mips']))
        self.assertEqual(['mips', 'i386'],
                         test(['-arch', 'mips', '-arch', 'i386']))

    def assertFlagsChanged(self, expected, flags):
        cmd = ['clang', '-c', 'source.c'] + flags
        opts = sut.classify_parameters(cmd)
        self.assertEqual(expected, opts['flags'])

    def assertFlagsUnchanged(self, flags):
        self.assertFlagsChanged(flags, flags)

    def assertFlagsFiltered(self, flags):
        self.assertFlagsChanged([], flags)

    def test_optimalizations_pass(self):
        self.assertFlagsUnchanged(['-O'])
        self.assertFlagsUnchanged(['-O1'])
        self.assertFlagsUnchanged(['-Os'])
        self.assertFlagsUnchanged(['-O2'])
        self.assertFlagsUnchanged(['-O3'])

    def test_include_pass(self):
        self.assertFlagsUnchanged([])
        self.assertFlagsUnchanged(['-include', '/usr/local/include'])
        self.assertFlagsUnchanged(['-I.'])
        self.assertFlagsUnchanged(['-I', '.'])
        self.assertFlagsUnchanged(['-I/usr/local/include'])
        self.assertFlagsUnchanged(['-I', '/usr/local/include'])
        self.assertFlagsUnchanged(['-I/opt', '-I', '/opt/otp/include'])
        self.assertFlagsUnchanged(['-isystem', '/path'])
        self.assertFlagsUnchanged(['-isystem=/path'])

    def test_define_pass(self):
        self.assertFlagsUnchanged(['-DNDEBUG'])
        self.assertFlagsUnchanged(['-UNDEBUG'])
        self.assertFlagsUnchanged(['-Dvar1=val1', '-Dvar2=val2'])
        self.assertFlagsUnchanged(['-Dvar="val ues"'])

    def test_output_filtered(self):
        self.assertFlagsFiltered(['-o', 'source.o'])

    def test_some_warning_filtered(self):
        self.assertFlagsFiltered(['-Wall'])
        self.assertFlagsFiltered(['-Wnoexcept'])
        self.assertFlagsFiltered(['-Wreorder', '-Wunused', '-Wundef'])
        self.assertFlagsUnchanged(['-Wno-reorder', '-Wno-unused'])

    def test_compile_only_flags_pass(self):
        self.assertFlagsUnchanged(['-std=C99'])
        self.assertFlagsUnchanged(['-nostdinc'])
        self.assertFlagsUnchanged(['-isystem', '/image/debian'])
        self.assertFlagsUnchanged(['-iprefix', '/usr/local'])
        self.assertFlagsUnchanged(['-iquote=me'])
        self.assertFlagsUnchanged(['-iquote', 'me'])

    def test_compile_and_link_flags_pass(self):
        self.assertFlagsUnchanged(['-fsinged-char'])
        self.assertFlagsUnchanged(['-fPIC'])
        self.assertFlagsUnchanged(['-stdlib=libc++'])
        self.assertFlagsUnchanged(['--sysroot', '/'])
        self.assertFlagsUnchanged(['-isysroot', '/'])

    def test_some_flags_filtered(self):
        self.assertFlagsFiltered(['-g'])
        self.assertFlagsFiltered(['-fsyntax-only'])
        self.assertFlagsFiltered(['-save-temps'])
        self.assertFlagsFiltered(['-init', 'my_init'])
        self.assertFlagsFiltered(['-sectorder', 'a', 'b', 'c'])


class Spy(object):
    def __init__(self):
        self.arg = None
        self.success = 0

    def call(self, params):
        self.arg = params
        return self.success


class RunAnalyzerTest(unittest.TestCase):

    @staticmethod
    def run_analyzer(content, failures_report):
        with libear.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.cpp')
            with open(filename, 'w') as handle:
                handle.write(content)

            opts = {
                'clang': 'clang',
                'directory': os.getcwd(),
                'flags': [],
                'direct_args': [],
                'file': filename,
                'output_dir': tmpdir,
                'output_format': 'plist',
                'output_failures': failures_report
            }
            spy = Spy()
            result = sut.run_analyzer(opts, spy.call)
            return (result, spy.arg)

    def test_run_analyzer(self):
        content = "int div(int n, int d) { return n / d; }"
        (result, fwds) = RunAnalyzerTest.run_analyzer(content, False)
        self.assertEqual(None, fwds)
        self.assertEqual(0, result['exit_code'])

    def test_run_analyzer_crash(self):
        content = "int div(int n, int d) { return n / d }"
        (result, fwds) = RunAnalyzerTest.run_analyzer(content, False)
        self.assertEqual(None, fwds)
        self.assertEqual(1, result['exit_code'])

    def test_run_analyzer_crash_and_forwarded(self):
        content = "int div(int n, int d) { return n / d }"
        (_, fwds) = RunAnalyzerTest.run_analyzer(content, True)
        self.assertEqual('crash', fwds['error_type'])
        self.assertEqual(1, fwds['exit_code'])
        self.assertTrue(len(fwds['error_output']) > 0)


class ReportFailureTest(unittest.TestCase):

    def assertUnderFailures(self, path):
        self.assertEqual('failures', os.path.basename(os.path.dirname(path)))

    def test_report_failure_create_files(self):
        with libear.TemporaryDirectory() as tmpdir:
            # create input file
            filename = os.path.join(tmpdir, 'test.c')
            with open(filename, 'w') as handle:
                handle.write('int main() { return 0')
            uname_msg = ' '.join(os.uname()) + os.linesep
            error_msg = 'this is my error output'
            # execute test
            opts = {
                'clang': 'clang',
                'directory': os.getcwd(),
                'flags': [],
                'file': filename,
                'output_dir': tmpdir,
                'language': 'c',
                'error_type': 'other_error',
                'error_output': error_msg,
                'exit_code': 13
            }
            sut.report_failure(opts)
            # verify the result
            result = dict()
            pp_file = None
            for root, _, files in os.walk(tmpdir):
                keys = [os.path.join(root, name) for name in files]
                for key in keys:
                    with open(key, 'r') as handle:
                        result[key] = handle.readlines()
                    if re.match(r'^(.*/)+clang(.*)\.i$', key):
                        pp_file = key

            # prepocessor file generated
            self.assertUnderFailures(pp_file)
            # info file generated and content dumped
            info_file = pp_file + '.info.txt'
            self.assertTrue(info_file in result)
            self.assertEqual('Other Error\n', result[info_file][1])
            self.assertEqual(uname_msg, result[info_file][3])
            # error file generated and content dumped
            error_file = pp_file + '.stderr.txt'
            self.assertTrue(error_file in result)
            self.assertEqual([error_msg], result[error_file])


class AnalyzerTest(unittest.TestCase):

    def test_nodebug_macros_appended(self):
        def test(flags):
            spy = Spy()
            opts = {'flags': flags, 'force_debug': True}
            self.assertEqual(spy.success,
                             sut.filter_debug_flags(opts, spy.call))
            return spy.arg['flags']

        self.assertEqual(['-UNDEBUG'], test([]))
        self.assertEqual(['-DNDEBUG', '-UNDEBUG'], test(['-DNDEBUG']))
        self.assertEqual(['-DSomething', '-UNDEBUG'], test(['-DSomething']))

    def test_set_language_fall_through(self):
        def language(expected, input):
            spy = Spy()
            input.update({'compiler': 'c', 'file': 'test.c'})
            self.assertEqual(spy.success, sut.language_check(input, spy.call))
            self.assertEqual(expected, spy.arg['language'])

        language('c',   {'language': 'c', 'flags': []})
        language('c++', {'language': 'c++', 'flags': []})

    def test_set_language_stops_on_not_supported(self):
        spy = Spy()
        input = {
            'compiler': 'c',
            'flags': [],
            'file': 'test.java',
            'language': 'java'
        }
        self.assertIsNone(sut.language_check(input, spy.call))
        self.assertIsNone(spy.arg)

    def test_set_language_sets_flags(self):
        def flags(expected, input):
            spy = Spy()
            input.update({'compiler': 'c', 'file': 'test.c'})
            self.assertEqual(spy.success, sut.language_check(input, spy.call))
            self.assertEqual(expected, spy.arg['flags'])

        flags(['-x', 'c'],   {'language': 'c', 'flags': []})
        flags(['-x', 'c++'], {'language': 'c++', 'flags': []})

    def test_set_language_from_filename(self):
        def language(expected, input):
            spy = Spy()
            input.update({'language': None, 'flags': []})
            self.assertEqual(spy.success, sut.language_check(input, spy.call))
            self.assertEqual(expected, spy.arg['language'])

        language('c',   {'file': 'file.c',   'compiler': 'c'})
        language('c++', {'file': 'file.c',   'compiler': 'c++'})
        language('c++', {'file': 'file.cxx', 'compiler': 'c'})
        language('c++', {'file': 'file.cxx', 'compiler': 'c++'})
        language('c++', {'file': 'file.cpp', 'compiler': 'c++'})
        language('c-cpp-output',   {'file': 'file.i', 'compiler': 'c'})
        language('c++-cpp-output', {'file': 'file.i', 'compiler': 'c++'})

    def test_arch_loop_sets_flags(self):
        def flags(archs):
            spy = Spy()
            input = {'flags': [], 'arch_list': archs}
            sut.arch_check(input, spy.call)
            return spy.arg['flags']

        self.assertEqual([], flags([]))
        self.assertEqual(['-arch', 'i386'], flags(['i386']))
        self.assertEqual(['-arch', 'i386'], flags(['i386', 'ppc']))
        self.assertEqual(['-arch', 'sparc'], flags(['i386', 'sparc']))

    def test_arch_loop_stops_on_not_supported(self):
        def stop(archs):
            spy = Spy()
            input = {'flags': [], 'arch_list': archs}
            self.assertIsNone(sut.arch_check(input, spy.call))
            self.assertIsNone(spy.arg)

        stop(['ppc'])
        stop(['ppc64'])


@sut.require([])
def method_without_expecteds(opts):
    return 0


@sut.require(['this', 'that'])
def method_with_expecteds(opts):
    return 0


@sut.require([])
def method_exception_from_inside(opts):
    raise Exception('here is one')


class RequireDecoratorTest(unittest.TestCase):

    def test_method_without_expecteds(self):
        self.assertEqual(method_without_expecteds(dict()), 0)
        self.assertEqual(method_without_expecteds({}), 0)
        self.assertEqual(method_without_expecteds({'this': 2}), 0)
        self.assertEqual(method_without_expecteds({'that': 3}), 0)

    def test_method_with_expecteds(self):
        self.assertRaises(KeyError, method_with_expecteds, dict())
        self.assertRaises(KeyError, method_with_expecteds, {})
        self.assertRaises(KeyError, method_with_expecteds, {'this': 2})
        self.assertRaises(KeyError, method_with_expecteds, {'that': 3})
        self.assertEqual(method_with_expecteds({'this': 0, 'that': 3}), 0)

    def test_method_exception_not_caught(self):
        self.assertRaises(Exception, method_exception_from_inside, dict())
