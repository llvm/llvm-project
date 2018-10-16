"""
                     The LLVM Compiler Infrastructure

This file is distributed under the University of Illinois Open Source
License. See LICENSE.TXT for details.

Provides the configuration class, which holds all information related to
how this invocation of the test suite should be run.
"""

from __future__ import absolute_import
from __future__ import print_function

# System modules
import os
import platform
import subprocess


# Third-party modules
import unittest2

# LLDB Modules
import lldbsuite


# The test suite.
suite = unittest2.TestSuite()

# The list of categories we said we care about
categoriesList = None
# set to true if we are going to use categories for cherry-picking test cases
useCategories = False
# Categories we want to skip
skipCategories = ["darwin-log"]
# use this to track per-category failures
failuresPerCategory = {}

# The path to LLDB.framework is optional.
lldbFrameworkPath = None

# Test suite repeat count.  Can be overwritten with '-# count'.
count = 1

# The 'arch' and 'compiler' can be specified via command line.
arch = None        # Must be initialized after option parsing
compiler = None    # Must be initialized after option parsing

# Path to the FileCheck testing tool. Not optional.
filecheck = None

# The arch might dictate some specific CFLAGS to be passed to the toolchain to build
# the inferior programs.  The global variable cflags_extras provides a hook to do
# just that.
cflags_extras = ''

# The filters (testclass.testmethod) used to admit tests into our test suite.
filters = []

# By default, we skip long running test case.  Use '-l' option to override.
skip_long_running_test = True

# Parsable mode silences headers, and any other output this script might generate, and instead
# prints machine-readable output similar to what clang tests produce.
parsable = False

# The regular expression pattern to match against eligible filenames as
# our test cases.
regexp = None

# Sets of tests which are excluded at runtime
skip_tests = None
xfail_tests = None

# By default, recorded session info for errored/failed test are dumped into its
# own file under a session directory named after the timestamp of the test suite
# run.  Use '-s session-dir-name' to specify a specific dir name.
sdir_name = None

# Valid options:
# f - test file name (without extension)
# n - test class name
# m - test method name
# a - architecture
# c - compiler path
# The default is to write all fields.
session_file_format = 'fnmac'

# Set this flag if there is any session info dumped during the test run.
sdir_has_content = False

# svn_info stores the output from 'svn info lldb.base.dir'.
svn_info = ''

# Default verbosity is 0.
verbose = 0

# By default, search from the script directory.
# We can't use sys.path[0] to determine the script directory
# because it doesn't work under a debugger
testdirs = [os.path.dirname(os.path.realpath(__file__))]

# Separator string.
separator = '-' * 70

failed = False

# LLDB Remote platform setting
lldb_platform_name = None
lldb_platform_url = None
lldb_platform_working_dir = None

# The base directory in which the tests are being built.
test_build_dir = None

# The only directory to scan for tests. If multiple test directories are
# specified, and an exclusive test subdirectory is specified, the latter option
# takes precedence.
exclusive_test_subdir = None

# Parallel execution settings
is_inferior_test_runner = False
num_threads = None
no_multiprocess_test_runner = False
test_runner_name = None

# Test results handling globals
results_filename = None
results_port = None
results_formatter_name = None
results_formatter_object = None
results_formatter_options = None
test_result = None

# Test rerun configuration vars
rerun_all_issues = False
rerun_max_file_threhold = 0

# The names of all tests. Used to assert we don't have two tests with the
# same base name.
all_tests = set()

def shouldSkipBecauseOfCategories(test_categories):
    if useCategories:
        if len(test_categories) == 0 or len(
                categoriesList & set(test_categories)) == 0:
            return True

    for category in skipCategories:
        if category in test_categories:
            return True

    return False


def get_absolute_path_to_exclusive_test_subdir():
    """
    If an exclusive test subdirectory is specified, return its absolute path.
    Otherwise return None.
    """
    test_directory = os.path.dirname(os.path.realpath(__file__))

    if not exclusive_test_subdir:
        return

    if len(exclusive_test_subdir) > 0:
        test_subdir = os.path.join(test_directory, exclusive_test_subdir)
        if os.path.isdir(test_subdir):
            return test_subdir

        print('specified test subdirectory {} is not a valid directory\n'
                .format(test_subdir))


def get_absolute_path_to_root_test_dir():
    """
    If an exclusive test subdirectory is specified, return its absolute path.
    Otherwise, return the absolute path of the root test directory.
    """
    test_subdir = get_absolute_path_to_exclusive_test_subdir()
    if test_subdir:
        return test_subdir

    return os.path.dirname(os.path.realpath(__file__))


def get_filecheck_path():
    """
    Get the path to the FileCheck testing tool.
    """
    if filecheck and os.path.lexists(filecheck):
        return filecheck
