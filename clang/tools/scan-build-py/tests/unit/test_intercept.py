# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libscanbuild.intercept as sut
from . import fixtures
import os.path


class InterceptUtilTest(fixtures.TestCase):

    def test_is_compiler_call_filter(self):
        def test(command):
            return sut.is_compiler_call({'command': [command]})

        self.assertTrue(test('clang'))
        self.assertTrue(test('clang-3.6'))
        self.assertTrue(test('clang++'))
        self.assertTrue(test('clang++-3.5.1'))
        self.assertTrue(test('cc'))
        self.assertTrue(test('c++'))
        self.assertTrue(test('gcc'))
        self.assertTrue(test('g++'))
        self.assertTrue(test('/usr/local/bin/gcc'))
        self.assertTrue(test('/usr/local/bin/g++'))
        self.assertTrue(test('/usr/local/bin/clang'))
        self.assertTrue(test('armv7_neno-linux-gnueabi-g++'))

        self.assertFalse(test(''))
        self.assertFalse(test('ld'))
        self.assertFalse(test('as'))
        self.assertFalse(test('/usr/local/bin/compiler'))

    def test_format_entry_filters_action(self):
        def test(command):
            return list(sut.format_entry(
                {'command': command, 'directory': '/opt/src/project'}))

        self.assertTrue(test(['cc', '-c', 'file.c', '-o', 'file.o']))
        self.assertFalse(test(['cc', '-E', 'file.c']))
        self.assertFalse(test(['cc', '-MM', 'file.c']))
        self.assertFalse(test(['cc', 'this.o', 'that.o', '-o', 'a.out']))
        self.assertFalse(test(['cc', '-print-prog-name']))

    def test_format_entry_normalize_filename(self):
        directory = os.path.join(os.sep, 'home', 'me', 'project')

        def test(command):
            result = list(sut.format_entry(
                {'command': command, 'directory': directory}))
            return result[0]['file']

        self.assertEqual(test(['cc', '-c', 'file.c']),
                         os.path.join(directory, 'file.c'))
        self.assertEqual(test(['cc', '-c', './file.c']),
                         os.path.join(directory, 'file.c'))
        self.assertEqual(test(['cc', '-c', '../file.c']),
                         os.path.join(os.path.dirname(directory), 'file.c'))
        self.assertEqual(test(['cc', '-c', '/opt/file.c']),
                         '/opt/file.c')

    def test_sip(self):
        def create_status_report(filename, message):
            content = """#!/usr/bin/env sh
                         echo 'sa-la-la-la'
                         echo 'la-la-la'
                         echo '{0}'
                         echo 'sa-la-la-la'
                         echo 'la-la-la'
                      """.format(message)
            lines = [line.strip() for line in content.split('\n')]
            with open(filename, 'w') as handle:
                handle.write('\n'.join(lines))
                handle.close()
            os.chmod(filename, 0x1ff)

        def create_csrutil(dest_dir, status):
            filename = os.path.join(dest_dir, 'csrutil')
            message = 'System Integrity Protection status: {0}'.format(status)
            return create_status_report(filename, message)

        def create_sestatus(dest_dir, status):
            filename = os.path.join(dest_dir, 'sestatus')
            message = 'SELinux status:\t{0}'.format(status)
            return create_status_report(filename, message)

        ENABLED = 'enabled'
        DISABLED = 'disabled'

        OSX = 'darwin'
        LINUX = 'linux'

        with fixtures.TempDir() as tmpdir:
            try:
                saved = os.environ['PATH']
                os.environ['PATH'] = tmpdir + ':' + saved

                create_csrutil(tmpdir, ENABLED)
                self.assertTrue(sut.is_preload_disabled(OSX))

                create_csrutil(tmpdir, DISABLED)
                self.assertFalse(sut.is_preload_disabled(OSX))

                create_sestatus(tmpdir, ENABLED)
                self.assertTrue(sut.is_preload_disabled(LINUX))

                create_sestatus(tmpdir, DISABLED)
                self.assertFalse(sut.is_preload_disabled(LINUX))
            finally:
                os.environ['PATH'] = saved

        try:
            saved = os.environ['PATH']
            os.environ['PATH'] = ''
            # shall be false when it's not in the path
            self.assertFalse(sut.is_preload_disabled(OSX))
            self.assertFalse(sut.is_preload_disabled(LINUX))

            self.assertFalse(sut.is_preload_disabled('unix'))
        finally:
            os.environ['PATH'] = saved
