# -*- coding: utf-8 -*-
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.

import libear
import libscanbuild.analyze as sut
import unittest

class ReportDirectoryTest(unittest.TestCase):

    # Test that successive report directory names ascend in lexicographic
    # order. This is required so that report directories from two runs of
    # scan-build can be easily matched up to compare results.
    def test_directory_name_comparison(self):
        with libear.TemporaryDirectory() as tmpdir, \
             sut.report_directory(tmpdir, False) as report_dir1, \
             sut.report_directory(tmpdir, False) as report_dir2, \
             sut.report_directory(tmpdir, False) as report_dir3:
            self.assertLess(report_dir1, report_dir2)
            self.assertLess(report_dir2, report_dir3)
