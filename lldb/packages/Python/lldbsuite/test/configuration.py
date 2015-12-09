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

def __setCrashInfoHook_Mac(text):
    from . import crashinfo
    crashinfo.setCrashReporterDescription(text)

def setupCrashInfoHook():
    if platform.system() == "Darwin":
        from . import lock
        test_dir = os.environ['LLDB_TEST']
        if not test_dir or not os.path.exists(test_dir):
            return
        dylib_lock = os.path.join(test_dir,"crashinfo.lock")
        dylib_src = os.path.join(test_dir,"crashinfo.c")
        dylib_dst = os.path.join(test_dir,"crashinfo.so")
        try:
            compile_lock = lock.Lock(dylib_lock)
            compile_lock.acquire()
            if not os.path.isfile(dylib_dst) or os.path.getmtime(dylib_dst) < os.path.getmtime(dylib_src):
                # we need to compile
                cmd = "SDKROOT= xcrun clang %s -o %s -framework Python -Xlinker -dylib -iframework /System/Library/Frameworks/ -Xlinker -F /System/Library/Frameworks/" % (dylib_src,dylib_dst)
                if subprocess.call(cmd,shell=True) != 0 or not os.path.isfile(dylib_dst):
                    raise Exception('command failed: "{}"'.format(cmd))
        finally:
            compile_lock.release()
            del compile_lock

        setCrashInfoHook = __setCrashInfoHook_Mac

    else:
        pass

# The test suite.
suite = unittest2.TestSuite()

dont_do_dsym_test = False
dont_do_dwarf_test = False
dont_do_dwo_test = False

# The list of categories we said we care about
categoriesList = None
# set to true if we are going to use categories for cherry-picking test cases
useCategories = False
# Categories we want to skip
skipCategories = []
# use this to track per-category failures
failuresPerCategory = {}

# The path to LLDB.framework is optional.
lldbFrameworkPath = None

# Test suite repeat count.  Can be overwritten with '-# count'.
count = 1

# The 'archs' and 'compilers' can be specified via command line.  The corresponding
# options can be specified more than once. For example, "-A x86_64 -A i386"
# => archs=['x86_64', 'i386'] and "-C gcc -C clang" => compilers=['gcc', 'clang'].
archs = None        # Must be initialized after option parsing
compilers = None    # Must be initialized after option parsing

# The arch might dictate some specific CFLAGS to be passed to the toolchain to build
# the inferior programs.  The global variable cflags_extras provides a hook to do
# just that.
cflags_extras = ''

# The breakpoint specification of the benchmark exe, as specified by the '-x' option.
bmBreakpointSpec = None
# The benchmark iteration count, as specified by the '-y' option.
bmIterationCount = -1

# The filters (testclass.testmethod) used to admit tests into our test suite.
filters = []

# By default, we do not skip build and cleanup.  Use '-S' option to override.
skip_build_and_cleanup = False

# By default, we skip long running test case.  Use '-l' option to override.
skip_long_running_test = True

# Parsable mode silences headers, and any other output this script might generate, and instead
# prints machine-readable output similar to what clang tests produce.
parsable = False

# The regular expression pattern to match against eligible filenames as our test cases.
regexp = None

# By default, tests are executed in place and cleanups are performed afterwards.
# Use '-r dir' option to relocate the tests and their intermediate files to a
# different directory and to forgo any cleanups.  The directory specified must
# not exist yet.
rdir = None

# By default, recorded session info for errored/failed test are dumped into its
# own file under a session directory named after the timestamp of the test suite
# run.  Use '-s session-dir-name' to specify a specific dir name.
sdir_name = None

# Set this flag if there is any session info dumped during the test run.
sdir_has_content = False

# svn_info stores the output from 'svn info lldb.base.dir'.
svn_info = ''

# svn_silent means do not try to obtain svn status
svn_silent = True

# Default verbosity is 0.
verbose = 1

# By default, search from the script directory.
# We can't use sys.path[0] to determine the script directory
# because it doesn't work under a debugger
testdirs = [ os.path.dirname(os.path.realpath(__file__)) ]

# Separator string.
separator = '-' * 70

failed = False

# LLDB Remote platform setting
lldb_platform_name = None
lldb_platform_url = None
lldb_platform_working_dir = None

# Parallel execution settings
is_inferior_test_runner = False
multiprocess_test_subdir = None
num_threads = None
output_on_success = False
no_multiprocess_test_runner = False
test_runner_name = None

# Test results handling globals
results_filename = None
results_port = None
results_formatter_name = None
results_formatter_object = None
results_formatter_options = None
test_result = None

# The names of all tests. Used to assert we don't have two tests with the same base name.
all_tests = set()

# safe default
setCrashInfoHook = lambda x : None

def shouldSkipBecauseOfCategories(test_categories):
    if useCategories:
        if len(test_categories) == 0 or len(categoriesList & set(test_categories)) == 0:
            return True

    for category in skipCategories:
        if category in test_categories:
            return True

    return False
