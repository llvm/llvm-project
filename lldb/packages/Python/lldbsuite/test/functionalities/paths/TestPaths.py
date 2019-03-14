"""
Test some lldb command abbreviations.
"""
from __future__ import print_function


import lldb
import os
import time
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestPaths(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_paths(self):
        '''Test to make sure no file names are set in the lldb.SBFileSpec objects returned by lldb.SBHostOS.GetLLDBPath() for paths that are directories'''
        dir_path_types = [lldb.ePathTypeLLDBShlibDir,
                          lldb.ePathTypeSupportExecutableDir,
                          lldb.ePathTypeHeaderDir,
                          lldb.ePathTypePythonDir,
                          lldb.ePathTypeLLDBSystemPlugins,
                          lldb.ePathTypeLLDBUserPlugins,
                          lldb.ePathTypeLLDBTempSystemDir,
                          lldb.ePathTypeClangDir]

        for path_type in dir_path_types:
            f = lldb.SBHostOS.GetLLDBPath(path_type)
            # No directory path types should have the filename set
            self.assertTrue(f.GetFilename() is None)

    @no_debug_info_test
    def test_directory_doesnt_end_with_slash(self):
        current_directory_spec = lldb.SBFileSpec(os.path.curdir)
        current_directory_string = current_directory_spec.GetDirectory()
        self.assertNotEqual(current_directory_string[-1:], '/')
        pass

    @skipUnlessPlatform(["windows"])
    @no_debug_info_test
    def test_windows_double_slash(self):
        '''Test to check the path with double slash is handled correctly '''
        # Create a path and see if lldb gets the directory and file right
        fspec = lldb.SBFileSpec("C:\\dummy1\\dummy2//unknown_file", True)
        self.assertEqual(
            os.path.normpath(
                fspec.GetDirectory()),
            os.path.normpath("C:/dummy1/dummy2"))
        self.assertEqual(fspec.GetFilename(), "unknown_file")
