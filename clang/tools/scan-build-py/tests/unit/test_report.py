# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libear
import libscanbuild.report as sut
import unittest
import os
import os.path


def run_bug_parse(content):
    with libear.TemporaryDirectory() as tmpdir:
        file_name = os.path.join(tmpdir, 'test.html')
        with open(file_name, 'w') as handle:
            handle.writelines(content)
        for bug in sut.parse_bug_html(file_name):
            return bug


def run_crash_parse(content, preproc):
    with libear.TemporaryDirectory() as tmpdir:
        file_name = os.path.join(tmpdir, preproc + '.info.txt')
        with open(file_name, 'w') as handle:
            handle.writelines(content)
        return sut.parse_crash(file_name)


class ParseFileTest(unittest.TestCase):

    def test_parse_bug(self):
        content = [
            "some header\n",
            "<!-- BUGDESC Division by zero -->\n",
            "<!-- BUGTYPE Division by zero -->\n",
            "<!-- BUGCATEGORY Logic error -->\n",
            "<!-- BUGFILE xx -->\n",
            "<!-- BUGLINE 5 -->\n",
            "<!-- BUGCOLUMN 22 -->\n",
            "<!-- BUGPATHLENGTH 4 -->\n",
            "<!-- BUGMETAEND -->\n",
            "<!-- REPORTHEADER -->\n",
            "some tails\n"]
        result = run_bug_parse(content)
        self.assertEqual(result['bug_category'], 'Logic error')
        self.assertEqual(result['bug_path_length'], 4)
        self.assertEqual(result['bug_line'], 5)
        self.assertEqual(result['bug_description'], 'Division by zero')
        self.assertEqual(result['bug_type'], 'Division by zero')
        self.assertEqual(result['bug_file'], 'xx')

    def test_parse_bug_empty(self):
        content = []
        result = run_bug_parse(content)
        self.assertEqual(result['bug_category'], 'Other')
        self.assertEqual(result['bug_path_length'], 1)
        self.assertEqual(result['bug_line'], 0)

    def test_parse_crash(self):
        content = [
            "/some/path/file.c\n",
            "Some very serious Error\n",
            "bla\n",
            "bla-bla\n"]
        result = run_crash_parse(content, 'file.i')
        self.assertEqual(result['source'], content[0].rstrip())
        self.assertEqual(result['problem'], content[1].rstrip())
        self.assertEqual(os.path.basename(result['file']),
                         'file.i')
        self.assertEqual(os.path.basename(result['info']),
                         'file.i.info.txt')
        self.assertEqual(os.path.basename(result['stderr']),
                         'file.i.stderr.txt')

    def test_parse_real_crash(self):
        import libscanbuild.runner as sut2
        import re
        with libear.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.c')
            with open(filename, 'w') as handle:
                handle.write('int main() { return 0')
            # produce failure report
            opts = {
                'clang': 'clang',
                'directory': os.getcwd(),
                'flags': [],
                'file': filename,
                'output_dir': tmpdir,
                'language': 'c',
                'error_type': 'other_error',
                'error_output': 'some output',
                'exit_code': 13
            }
            sut2.report_failure(opts)
            # find the info file
            pp_file = None
            for root, _, files in os.walk(tmpdir):
                keys = [os.path.join(root, name) for name in files]
                for key in keys:
                    if re.match(r'^(.*/)+clang(.*)\.i$', key):
                        pp_file = key
            self.assertIsNot(pp_file, None)
            # read the failure report back
            result = sut.parse_crash(pp_file + '.info.txt')
            self.assertEqual(result['source'], filename)
            self.assertEqual(result['problem'], 'Other Error')
            self.assertEqual(result['file'], pp_file)
            self.assertEqual(result['info'], pp_file + '.info.txt')
            self.assertEqual(result['stderr'], pp_file + '.stderr.txt')


class ReportMethodTest(unittest.TestCase):

    def test_chop(self):
        self.assertEqual('file', sut.chop('/prefix', '/prefix/file'))
        self.assertEqual('file', sut.chop('/prefix/', '/prefix/file'))
        self.assertEqual('lib/file', sut.chop('/prefix/', '/prefix/lib/file'))
        self.assertEqual('/prefix/file', sut.chop('', '/prefix/file'))

    def test_chop_when_cwd(self):
        self.assertEqual('../src/file', sut.chop('/cwd', '/src/file'))
        self.assertEqual('../src/file', sut.chop('/prefix/cwd',
                                                 '/prefix/src/file'))


class GetPrefixFromCompilationDatabaseTest(unittest.TestCase):

    def test_with_different_filenames(self):
        self.assertEqual(
            sut.commonprefix(['/tmp/a.c', '/tmp/b.c']), '/tmp')

    def test_with_different_dirnames(self):
        self.assertEqual(
            sut.commonprefix(['/tmp/abs/a.c', '/tmp/ack/b.c']), '/tmp')

    def test_no_common_prefix(self):
        self.assertEqual(
            sut.commonprefix(['/tmp/abs/a.c', '/usr/ack/b.c']), '/')

    def test_with_single_file(self):
        self.assertEqual(
            sut.commonprefix(['/tmp/a.c']), '/tmp')

    def test_empty(self):
        self.assertEqual(
            sut.commonprefix([]), '')
