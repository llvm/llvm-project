"""
A simple testing framework for lldb using python's unit testing framework.

Tests for lldb are written as python scripts which take advantage of the script
bridging provided by LLDB.framework to interact with lldb core.

A specific naming pattern is followed by the .py script to be recognized as
a module which implements a test scenario, namely, Test*.py.

To specify the directories where "Test*.py" python test scripts are located,
you need to pass in a list of directory names.  By default, the current
working directory is searched if nothing is specified on the command line.

Type:

./dotest.py -h

for available options.
"""

from __future__ import absolute_import
from __future__ import print_function

# System modules
import atexit
import importlib
import os
import errno
import platform
import progress
import signal
import socket
import subprocess
import sys
import inspect

# Third-party modules
import six
import unittest2

# LLDB Modules
import lldbsuite
from . import dotest_args
from . import lldbtest_config
from . import test_categories
from . import test_results
from .test_results import EventBuilder
from ..support import seven

def is_exe(fpath):
    """Returns true if fpath is an executable."""
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def which(program):
    """Returns the full path to a program; None otherwise."""
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy 'writeln' method"""
    def __init__(self,stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ('stream', '__getstate__'):
            raise AttributeError(attr)
        return getattr(self.stream,attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write('\n') # text-mode streams translate to \r\n if needed

#
# Global variables:
#

# The test suite.
suite = unittest2.TestSuite()

# By default, benchmarks tests are not run.
just_do_benchmarks_test = False

dont_do_dsym_test = False
dont_do_dwarf_test = False
dont_do_dwo_test = False

# The blacklist is optional (-b blacklistFile) and allows a central place to skip
# testclass's and/or testclass.testmethod's.
blacklist = None

# The dictionary as a result of sourcing blacklistFile.
blacklistConfig = {}

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

# The config file is optional.
configFile = None

# Test suite repeat count.  Can be overwritten with '-# count'.
count = 1

# The dictionary as a result of sourcing configFile.
config = {}
# The pre_flight and post_flight functions come from reading a config file.
pre_flight = None
post_flight = None
# So do the lldbtest_remote_sandbox and lldbtest_remote_shell_template variables.
lldbtest_remote_sandbox = None
lldbtest_remote_shell_template = None

# The 'archs' and 'compilers' can be specified via either command line or configFile,
# with the command line overriding the configFile.  The corresponding options can be
# specified more than once. For example, "-A x86_64 -A i386" => archs=['x86_64', 'i386']
# and "-C gcc -C clang" => compilers=['gcc', 'clang'].
archs = None        # Must be initialized after option parsing
compilers = None    # Must be initialized after option parsing

# The arch might dictate some specific CFLAGS to be passed to the toolchain to build
# the inferior programs.  The global variable cflags_extras provides a hook to do
# just that.
cflags_extras = ''

# Dump the Python sys.path variable.  Use '-D' to dump sys.path.
dumpSysPath = False

# Full path of the benchmark executable, as specified by the '-e' option.
bmExecutable = None
# The breakpoint specification of bmExecutable, as specified by the '-x' option.
bmBreakpointSpec = None
# The benchmark iteration count, as specified by the '-y' option.
bmIterationCount = -1

# By default, don't exclude any directories.  Use '-X' to add one excluded directory.
excluded = set(['.svn', '.git'])

# By default, failfast is False.  Use '-F' to overwrite it.
failfast = False

# The filters (testclass.testmethod) used to admit tests into our test suite.
filters = []

# The runhooks is a list of lldb commands specifically for the debugger.
# Use '-k' to specify a runhook.
runHooks = []

# If '-g' is specified, the filterspec is not exclusive.  If a test module does
# not contain testclass.testmethod which matches the filterspec, the whole test
# module is still admitted into our test suite.  fs4all flag defaults to True.
fs4all = True

# Ignore the build search path relative to this script to locate the lldb.py module.
ignore = False

# By default, we do not skip build and cleanup.  Use '-S' option to override.
skip_build_and_cleanup = False

# By default, we skip long running test case.  Use '-l' option to override.
skip_long_running_test = True

# By default, we print the build dir, lldb version, and svn info.  Use '-n' option to
# turn it off.
noHeaders = False

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

# Set to True only if verbose is 0 and LLDB trace mode is off.
progress_bar = False

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
results_file_object = None
results_formatter_name = None
results_formatter_object = None
results_formatter_options = None

# The names of all tests. Used to assert we don't have two tests with the same base name.
all_tests = set()

def usage(parser):
    parser.print_help()
    if verbose > 0:
        print("""
Examples:

This is an example of using the -f option to pinpoint to a specific test class
and test method to be run:

$ ./dotest.py -f ClassTypesTestCase.test_with_dsym_and_run_command
----------------------------------------------------------------------
Collected 1 test

test_with_dsym_and_run_command (TestClassTypes.ClassTypesTestCase)
Test 'frame variable this' when stopped on a class constructor. ... ok

----------------------------------------------------------------------
Ran 1 test in 1.396s

OK

And this is an example of using the -p option to run a single file (the filename
matches the pattern 'ObjC' and it happens to be 'TestObjCMethods.py'):

$ ./dotest.py -v -p ObjC
----------------------------------------------------------------------
Collected 4 tests

test_break_with_dsym (TestObjCMethods.FoundationTestCase)
Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'. ... ok
test_break_with_dwarf (TestObjCMethods.FoundationTestCase)
Test setting objc breakpoints using '_regexp-break' and 'breakpoint set'. ... ok
test_data_type_and_expr_with_dsym (TestObjCMethods.FoundationTestCase)
Lookup objective-c data types and evaluate expressions. ... ok
test_data_type_and_expr_with_dwarf (TestObjCMethods.FoundationTestCase)
Lookup objective-c data types and evaluate expressions. ... ok

----------------------------------------------------------------------
Ran 4 tests in 16.661s

OK

Running of this script also sets up the LLDB_TEST environment variable so that
individual test cases can locate their supporting files correctly.  The script
tries to set up Python's search paths for modules by looking at the build tree
relative to this script.  See also the '-i' option in the following example.

Finally, this is an example of using the lldb.py module distributed/installed by
Xcode4 to run against the tests under the 'forward' directory, and with the '-w'
option to add some delay between two tests.  It uses ARCH=x86_64 to specify that
as the architecture and CC=clang to specify the compiler used for the test run:

$ PYTHONPATH=/Xcode4/Library/PrivateFrameworks/LLDB.framework/Versions/A/Resources/Python ARCH=x86_64 CC=clang ./dotest.py -v -w -i forward

Session logs for test failures/errors will go into directory '2010-11-11-13_56_16'
----------------------------------------------------------------------
Collected 2 tests

test_with_dsym_and_run_command (TestForwardDeclaration.ForwardDeclarationTestCase)
Display *bar_ptr when stopped on a function with forward declaration of struct bar. ... ok
test_with_dwarf_and_run_command (TestForwardDeclaration.ForwardDeclarationTestCase)
Display *bar_ptr when stopped on a function with forward declaration of struct bar. ... ok

----------------------------------------------------------------------
Ran 2 tests in 5.659s

OK

The 'Session ...' verbiage is recently introduced (see also the '-s' option) to
notify the directory containing the session logs for test failures or errors.
In case there is any test failure/error, a similar message is appended at the
end of the stderr output for your convenience.

ENABLING LOGS FROM TESTS

Option 1:

Writing logs into different files per test case::

This option is particularly useful when multiple dotest instances are created
by dosep.py

$ ./dotest.py --channel "lldb all"

$ ./dotest.py --channel "lldb all" --channel "gdb-remote packets"

These log files are written to:

<session-dir>/<test-id>-host.log (logs from lldb host process)
<session-dir>/<test-id>-server.log (logs from debugserver/lldb-server)
<session-dir>/<test-id>-<test-result>.log (console logs)

By default, logs from successful runs are deleted.  Use the --log-success flag
to create reference logs for debugging.

$ ./dotest.py --log-success

Option 2: (DEPRECATED)

The following options can only enable logs from the host lldb process.
Only categories from the "lldb" or "gdb-remote" channels can be enabled
They also do not automatically enable logs in locally running debug servers.
Also, logs from all test case are written into each log file

o LLDB_LOG: if defined, specifies the log file pathname for the 'lldb' subsystem
  with a default option of 'event process' if LLDB_LOG_OPTION is not defined.

o GDB_REMOTE_LOG: if defined, specifies the log file pathname for the
  'process.gdb-remote' subsystem with a default option of 'packets' if
  GDB_REMOTE_LOG_OPTION is not defined.

""")
    sys.exit(0)


def setCrashInfoHook_Mac(text):
    from . import crashinfo
    crashinfo.setCrashReporterDescription(text)

# implement this in some suitable way for your platform, and then bind it
# to setCrashInfoHook
def setCrashInfoHook_NonMac(text):
    pass

setCrashInfoHook = None

def deleteCrashInfoDylib(dylib_path):
    try:
        # Need to modify this to handle multiple tests running at the same time.  If we move this
        # to the test's real dir, all should be we run sequentially within a test directory.
        # os.remove(dylib_path)
        None
    finally:
        pass

def setupCrashInfoHook():
    global setCrashInfoHook
    setCrashInfoHook = setCrashInfoHook_NonMac # safe default
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

        setCrashInfoHook = setCrashInfoHook_Mac

    else:
        pass

def shouldSkipBecauseOfCategories(test_categories):
    global useCategories, categoriesList, skipCategories

    if useCategories:
        if len(test_categories) == 0 or len(categoriesList & set(test_categories)) == 0:
            return True

    for category in skipCategories:
        if category in test_categories:
            return True

    return False

def parseOptionsAndInitTestdirs():
    """Initialize the list of directories containing our unittest scripts.

    '-h/--help as the first option prints out usage info and exit the program.
    """

    global just_do_benchmarks_test
    global dont_do_dsym_test
    global dont_do_dwarf_test
    global dont_do_dwo_test
    global blacklist
    global blacklistConfig
    global categoriesList
    global validCategories
    global useCategories
    global skipCategories
    global lldbFrameworkPath
    global configFile
    global archs
    global compilers
    global count
    global dumpSysPath
    global bmExecutable
    global bmBreakpointSpec
    global bmIterationCount
    global failfast
    global filters
    global fs4all
    global ignore
    global progress_bar
    global runHooks
    global skip_build_and_cleanup
    global skip_long_running_test
    global noHeaders
    global parsable
    global regexp
    global rdir
    global sdir_name
    global svn_silent
    global verbose
    global testdirs
    global lldb_platform_name
    global lldb_platform_url
    global lldb_platform_working_dir
    global setCrashInfoHook
    global is_inferior_test_runner
    global multiprocess_test_subdir
    global num_threads
    global output_on_success
    global no_multiprocess_test_runner
    global test_runner_name
    global results_filename
    global results_formatter_name
    global results_formatter_options
    global results_port

    do_help = False

    platform_system = platform.system()
    platform_machine = platform.machine()

    parser = dotest_args.create_parser()
    args = dotest_args.parse_args(parser, sys.argv[1:])

    if args.unset_env_varnames:
        for env_var in args.unset_env_varnames:
            if env_var in os.environ:
                # From Python Doc: When unsetenv() is supported, deletion of items in os.environ
                # is automatically translated into a corresponding call to unsetenv().
                del os.environ[env_var]
                #os.unsetenv(env_var)

    if args.set_env_vars:
        for env_var in args.set_env_vars:
            parts = env_var.split('=', 1)
            if len(parts) == 1:
                os.environ[parts[0]] = ""
            else:
                os.environ[parts[0]] = parts[1]

    # only print the args if being verbose (and parsable is off)
    if args.v and not args.q:
        print(sys.argv)

    if args.h:
        do_help = True

    if args.compilers:
        compilers = args.compilers
    else:
        # Use a compiler appropriate appropriate for the Apple SDK if one was specified
        if platform_system == 'Darwin' and args.apple_sdk:
            compilers = [seven.get_command_output('xcrun -sdk "%s" -find clang 2> /dev/null' % (args.apple_sdk))]
        else:
            # 'clang' on ubuntu 14.04 is 3.4 so we try clang-3.5 first
            candidateCompilers = ['clang-3.5', 'clang', 'gcc']
            for candidate in candidateCompilers:
                if which(candidate):
                    compilers = [candidate]
                    break

    if args.channels:
        lldbtest_config.channels = args.channels

    if args.log_success:
        lldbtest_config.log_success = args.log_success

    # Set SDKROOT if we are using an Apple SDK
    if platform_system == 'Darwin' and args.apple_sdk:
        os.environ['SDKROOT'] = seven.get_command_output('xcrun --sdk "%s" --show-sdk-path 2> /dev/null' % (args.apple_sdk))

    if args.archs:
        archs = args.archs
        for arch in archs:
            if arch.startswith('arm') and platform_system == 'Darwin' and not args.apple_sdk:
                os.environ['SDKROOT'] = seven.get_command_output('xcrun --sdk iphoneos.internal --show-sdk-path 2> /dev/null')
                if not os.path.exists(os.environ['SDKROOT']):
                    os.environ['SDKROOT'] = seven.get_command_output('xcrun --sdk iphoneos --show-sdk-path 2> /dev/null')
    else:
        archs = [platform_machine]

    if args.categoriesList:
        categoriesList = set(test_categories.validate(args.categoriesList, False))
        useCategories = True
    else:
        categoriesList = []

    if args.skipCategories:
        skipCategories = test_categories.validate(args.skipCategories, False)

    if args.D:
        dumpSysPath = True

    if args.E:
        cflags_extras = args.E
        os.environ['CFLAGS_EXTRAS'] = cflags_extras

    # argparse makes sure we have correct options
    if args.N == 'dwarf':
        dont_do_dwarf_test = True
    elif args.N == 'dwo':
        dont_do_dwo_test = True
    elif args.N == 'dsym':
        dont_do_dsym_test = True

    if args.a or args.plus_a:
        print("Options '-a' and '+a' have been deprecated. Please use the test category\n"
              "functionality (-G pyapi, --skip-category pyapi) instead.")
        sys.exit(1)

    if args.m or args.plus_m:
        print("Options '-m' and '+m' have been deprecated. Please use the test category\n"
              "functionality (-G lldb-mi, --skip-category lldb-mi) instead.")
        sys.exit(1)

    if args.plus_b:
        just_do_benchmarks_test = True

    if args.b:
        if args.b.startswith('-'):
            usage(parser)
        blacklistFile = args.b
        if not os.path.isfile(blacklistFile):
            print('Blacklist file:', blacklistFile, 'does not exist!')
            usage(parser)
        # Now read the blacklist contents and assign it to blacklist.
        execfile(blacklistFile, globals(), blacklistConfig)
        blacklist = blacklistConfig.get('blacklist')

    if args.c:
        if args.c.startswith('-'):
            usage(parser)
        configFile = args.c
        if not os.path.isfile(configFile):
            print('Config file:', configFile, 'does not exist!')
            usage(parser)

    if args.d:
        sys.stdout.write("Suspending the process %d to wait for debugger to attach...\n" % os.getpid())
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGSTOP)

    if args.e:
        if args.e.startswith('-'):
            usage(parser)
        bmExecutable = args.e
        if not is_exe(bmExecutable):
            usage(parser)

    if args.F:
        failfast = True

    if args.f:
        if any([x.startswith('-') for x in args.f]):
            usage(parser)
        filters.extend(args.f)
        # Shut off multiprocessing mode when additional filters are specified.
        # The rational is that the user is probably going after a very specific
        # test and doesn't need a bunch of parallel test runners all looking for
        # it in a frenzy.  Also, '-v' now spits out all test run output even
        # on success, so the standard recipe for redoing a failing test (with -v
        # and a -f to filter to the specific test) now causes all test scanning
        # (in parallel) to print results for do-nothing runs in a very distracting
        # manner.  If we really need filtered parallel runs in the future, consider
        # adding a --no-output-on-success that prevents -v from setting
        # output-on-success.
        no_multiprocess_test_runner = True

    if args.g:
        fs4all = False

    if args.i:
        ignore = True

    if args.k:
        runHooks.extend(args.k)

    if args.l:
        skip_long_running_test = False

    if args.framework:
        lldbFrameworkPath = args.framework

    if args.executable:
        lldbtest_config.lldbExec = args.executable

    if args.libcxx:
        os.environ["LIBCXX_PATH"] = args.libcxx

    if args.n:
        noHeaders = True

    if args.p:
        if args.p.startswith('-'):
            usage(parser)
        regexp = args.p

    if args.q:
        noHeaders = True
        parsable = True

    if args.P and not args.v:
        progress_bar = True
        verbose = 0

    if args.R:
        if args.R.startswith('-'):
            usage(parser)
        rdir = os.path.abspath(args.R)
        if os.path.exists(rdir):
            import shutil
            print('Removing tree:', rdir)
            shutil.rmtree(rdir)

    if args.r:
        if args.r.startswith('-'):
            usage(parser)
        rdir = os.path.abspath(args.r)
        if os.path.exists(rdir):
            print('Relocated directory:', rdir, 'must not exist!')
            usage(parser)

    if args.S:
        skip_build_and_cleanup = True

    if args.s:
        if args.s.startswith('-'):
            usage(parser)
        sdir_name = args.s

    if args.t:
        os.environ['LLDB_COMMAND_TRACE'] = 'YES'

    if args.T:
        svn_silent = False

    if args.v:
        verbose = 2

    if args.w:
        os.environ['LLDB_WAIT_BETWEEN_TEST_CASES'] = 'YES'

    if args.X:
        if args.X.startswith('-'):
            usage(parser)
        excluded.add(args.X)

    if args.x:
        if args.x.startswith('-'):
            usage(parser)
        bmBreakpointSpec = args.x

    # argparse makes sure we have a number
    if args.y:
        bmIterationCount = args.y

    # argparse makes sure we have a number
    if args.sharp:
        count = args.sharp

    if sys.platform.startswith('win32'):
        os.environ['LLDB_DISABLE_CRASH_DIALOG'] = str(args.disable_crash_dialog)
        os.environ['LLDB_LAUNCH_INFERIORS_WITHOUT_CONSOLE'] = str(args.hide_inferior_console)

    if do_help == True:
        usage(parser)

    if args.no_multiprocess:
        no_multiprocess_test_runner = True

    if args.inferior:
        is_inferior_test_runner = True

    # Turn on output_on_sucess if either explicitly added or -v specified.
    if args.output_on_success or args.v:
        output_on_success = True

    if args.num_threads:
        num_threads = args.num_threads

    if args.test_subdir:
        multiprocess_test_subdir = args.test_subdir

    if args.test_runner_name:
        test_runner_name = args.test_runner_name

    # Capture test results-related args.
    if args.curses and not args.inferior:
        # Act as if the following args were set.
        args.results_formatter = "lldbsuite.test.curses_results.Curses"
        args.results_file = "stdout"

    if args.results_file:
        results_filename = args.results_file

    if args.results_port:
        results_port = args.results_port

    if args.results_file and args.results_port:
        sys.stderr.write(
            "only one of --results-file and --results-port should "
            "be specified\n")
        usage(args)

    if args.results_formatter:
        results_formatter_name = args.results_formatter
    if args.results_formatter_options:
        results_formatter_options = args.results_formatter_options

    if args.lldb_platform_name:
        lldb_platform_name = args.lldb_platform_name
    if args.lldb_platform_url:
        lldb_platform_url = args.lldb_platform_url
    if args.lldb_platform_working_dir:
        lldb_platform_working_dir = args.lldb_platform_working_dir

    if args.event_add_entries and len(args.event_add_entries) > 0:
        entries = {}
        # Parse out key=val pairs, separated by comma
        for keyval in args.event_add_entries.split(","):
            key_val_entry = keyval.split("=")
            if len(key_val_entry) == 2:
                (key, val) = key_val_entry
                val_parts = val.split(':')
                if len(val_parts) > 1:
                    (val, val_type) = val_parts
                    if val_type == 'int':
                        val = int(val)
                entries[key] = val
        # Tell the event builder to create all events with these
        # key/val pairs in them.
        if len(entries) > 0:
            test_results.EventBuilder.add_entries_to_all_events(entries)

    # Gather all the dirs passed on the command line.
    if len(args.args) > 0:
        testdirs = list(map(os.path.abspath, args.args))
        # Shut off multiprocessing mode when test directories are specified.
        no_multiprocess_test_runner = True

    # If '-r dir' is specified, the tests should be run under the relocated
    # directory.  Let's copy the testdirs over.
    if rdir:
        from shutil import copytree, ignore_patterns

        tmpdirs = []
        orig_testdirs = testdirs[:]
        for srcdir in testdirs:
            # For example, /Volumes/data/lldb/svn/ToT/test/functionalities/watchpoint/hello_watchpoint
            # shall be split into ['/Volumes/data/lldb/svn/ToT/', 'functionalities/watchpoint/hello_watchpoint'].
            # Utilize the relative path to the 'test' directory to make our destination dir path.
            if ("test" + os.sep) in srcdir:
                to_split_on = "test" + os.sep
            else:
                to_split_on = "test"
            dstdir = os.path.join(rdir, srcdir.split(to_split_on)[1])
            dstdir = dstdir.rstrip(os.sep)
            # Don't copy the *.pyc and .svn stuffs.
            copytree(srcdir, dstdir, ignore=ignore_patterns('*.pyc', '.svn'))
            tmpdirs.append(dstdir)

        # This will be our modified testdirs.
        testdirs = tmpdirs

        # With '-r dir' specified, there's no cleanup of intermediate test files.
        os.environ["LLDB_DO_CLEANUP"] = 'NO'

        # If the original testdirs is ['test'], the make directory has already been copied
        # recursively and is contained within the rdir/test dir.  For anything
        # else, we would need to copy over the make directory and its contents,
        # so that, os.listdir(rdir) looks like, for example:
        #
        #     array_types conditional_break make
        #
        # where the make directory contains the Makefile.rules file.
        if len(testdirs) != 1 or os.path.basename(orig_testdirs[0]) != 'test':
            scriptdir = os.path.dirname(__file__)
            # Don't copy the .svn stuffs.
            copytree(os.path.join(scriptdir, 'make'), os.path.join(rdir, 'make'),
                     ignore=ignore_patterns('.svn'))

    #print("testdirs:", testdirs)

    # Source the configFile if specified.
    # The side effect, if any, will be felt from this point on.  An example
    # config file may be these simple two lines:
    #
    # sys.stderr = open("/tmp/lldbtest-stderr", "w")
    # sys.stdout = open("/tmp/lldbtest-stdout", "w")
    #
    # which will reassign the two file objects to sys.stderr and sys.stdout,
    # respectively.
    #
    # See also lldb-trunk/examples/test/usage-config.
    global config, pre_flight, post_flight, lldbtest_remote_sandbox, lldbtest_remote_shell_template
    if configFile:
        # Pass config (a dictionary) as the locals namespace for side-effect.
        execfile(configFile, globals(), config)
        #print("config:", config)
        if "pre_flight" in config:
            pre_flight = config["pre_flight"]
            if not six.callable(pre_flight):
                print("fatal error: pre_flight is not callable, exiting.")
                sys.exit(1)
        if "post_flight" in config:
            post_flight = config["post_flight"]
            if not six.callable(post_flight):
                print("fatal error: post_flight is not callable, exiting.")
                sys.exit(1)
        if "lldbtest_remote_sandbox" in config:
            lldbtest_remote_sandbox = config["lldbtest_remote_sandbox"]
        if "lldbtest_remote_shell_template" in config:
            lldbtest_remote_shell_template = config["lldbtest_remote_shell_template"]
        #print("sys.stderr:", sys.stderr)
        #print("sys.stdout:", sys.stdout)

def getXcodeOutputPaths(lldbRootDirectory):
    result = []

    # These are for xcode build directories.
    xcode3_build_dir = ['build']
    xcode4_build_dir = ['build', 'lldb', 'Build', 'Products']

    configurations = [['Debug'], ['DebugClang'], ['Release'], ['BuildAndIntegration']]
    xcode_build_dirs = [xcode3_build_dir, xcode4_build_dir]
    for configuration in configurations:
        for xcode_build_dir in xcode_build_dirs:
            outputPath = os.path.join(lldbRootDirectory, *(xcode_build_dir+configuration) )
            result.append(outputPath)

    return result


def createSocketToLocalPort(port):
    def socket_closer(s):
        """Close down an opened socket properly."""
        s.shutdown(socket.SHUT_RDWR)
        s.close()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("localhost", port))
    return (sock, lambda: socket_closer(sock))


def setupTestResults():
    """Sets up test results-related objects based on arg settings."""
    global results_filename
    global results_file_object
    global results_formatter_name
    global results_formatter_object
    global results_formatter_options
    global results_port

    default_formatter_name = None
    cleanup_func = None

    if results_filename:
        # Open the results file for writing.
        if results_filename == 'stdout':
            results_file_object = sys.stdout
            cleanup_func = None
        elif results_filename == 'stderr':
            results_file_object = sys.stderr
            cleanup_func = None
        else:
            results_file_object = open(results_filename, "w")
            cleanup_func = results_file_object.close
        default_formatter_name = "lldbsuite.test.test_results.XunitFormatter"
    elif results_port:
        # Connect to the specified localhost port.
        results_file_object, cleanup_func = createSocketToLocalPort(
            results_port)
        default_formatter_name = "lldbsuite.test.test_results.RawPickledFormatter"

    if results_file_object:
        # We care about the formatter.  Choose user-specified or, if
        # none specified, use the default for the output type.
        if results_formatter_name:
            formatter_name = results_formatter_name
        else:
            formatter_name = default_formatter_name

        # Create an instance of the class.  First figure out the package/module.
        components = formatter_name.split(".")
        module = importlib.import_module(".".join(components[:-1]))

        # Create the class name we need to load.
        clazz = getattr(module, components[-1])

        # Handle formatter options for the results formatter class.
        formatter_arg_parser = clazz.arg_parser()
        if results_formatter_options and len(results_formatter_options) > 0:
            command_line_options = results_formatter_options
        else:
            command_line_options = []

        formatter_options = formatter_arg_parser.parse_args(
            command_line_options)

        # Create the TestResultsFormatter given the processed options.
        results_formatter_object = clazz(results_file_object, formatter_options)

        # Start the results formatter session - we'll only have one
        # during a given dotest process invocation.
        initialize_event = EventBuilder.bare_event("initialize")
        if isMultiprocessTestRunner():
            if test_runner_name is not None and test_runner_name == "serial":
                # Only one worker queue here.
                worker_count = 1
            else:
                # Workers will be the number of threads specified.
                worker_count = num_threads
        else:
            worker_count = 1
        initialize_event["worker_count"] = worker_count

        results_formatter_object.handle_event(initialize_event)

        def shutdown_formatter():
            # Tell the formatter to write out anything it may have
            # been saving until the very end (e.g. xUnit results
            # can't complete its output until this point).
            results_formatter_object.send_terminate_as_needed()

            # And now close out the output file-like object.
            if cleanup_func is not None:
                cleanup_func()

        atexit.register(shutdown_formatter)


def getOutputPaths(lldbRootDirectory):
    """
    Returns typical build output paths for the lldb executable

    lldbDirectory - path to the root of the lldb svn/git repo
    """
    result = []

    if sys.platform == 'darwin':
        result.extend(getXcodeOutputPaths(lldbRootDirectory))

    # cmake builds?  look for build or build/host folder next to llvm directory
    # lldb is located in llvm/tools/lldb so we need to go up three levels
    llvmParentDir = os.path.abspath(os.path.join(lldbRootDirectory, os.pardir, os.pardir, os.pardir))
    result.append(os.path.join(llvmParentDir, 'build', 'bin'))
    result.append(os.path.join(llvmParentDir, 'build', 'host', 'bin'))

    # some cmake developers keep their build directory beside their lldb directory
    lldbParentDir = os.path.abspath(os.path.join(lldbRootDirectory, os.pardir))
    result.append(os.path.join(lldbParentDir, 'build', 'bin'))
    result.append(os.path.join(lldbParentDir, 'build', 'host', 'bin'))

    return result

def setupSysPath():
    """
    Add LLDB.framework/Resources/Python to the search paths for modules.
    As a side effect, we also discover the 'lldb' executable and export it here.
    """

    global rdir
    global testdirs
    global dumpSysPath
    global noHeaders
    global svn_info
    global svn_silent
    global lldbFrameworkPath

    # Get the directory containing the current script.
    if "DOTEST_PROFILE" in os.environ and "DOTEST_SCRIPT_DIR" in os.environ:
        scriptPath = os.environ["DOTEST_SCRIPT_DIR"]
    else:
        scriptPath = os.path.dirname(os.path.realpath(__file__))
    if not scriptPath.endswith('test'):
        print("This script expects to reside in lldb's test directory.")
        sys.exit(-1)

    if rdir:
        # Set up the LLDB_TEST environment variable appropriately, so that the
        # individual tests can be located relatively.
        #
        # See also lldbtest.TestBase.setUpClass(cls).
        if len(testdirs) == 1 and os.path.basename(testdirs[0]) == 'test':
            os.environ["LLDB_TEST"] = os.path.join(rdir, 'test')
        else:
            os.environ["LLDB_TEST"] = rdir
    else:
        os.environ["LLDB_TEST"] = scriptPath

    # Set up the LLDB_SRC environment variable, so that the tests can locate
    # the LLDB source code.
    os.environ["LLDB_SRC"] = lldbsuite.lldb_root

    pluginPath = os.path.join(scriptPath, 'plugins')
    toolsLLDBMIPath = os.path.join(scriptPath, 'tools', 'lldb-mi')
    toolsLLDBServerPath = os.path.join(scriptPath, 'tools', 'lldb-server')

    # Insert script dir, plugin dir, lldb-mi dir and lldb-server dir to the sys.path.
    sys.path.insert(0, pluginPath)
    sys.path.insert(0, toolsLLDBMIPath)      # Adding test/tools/lldb-mi to the path makes it easy
                                             # to "import lldbmi_testcase" from the MI tests
    sys.path.insert(0, toolsLLDBServerPath)  # Adding test/tools/lldb-server to the path makes it easy
                                             # to "import lldbgdbserverutils" from the lldb-server tests

    # This is the root of the lldb git/svn checkout
    # When this changes over to a package instead of a standalone script, this
    # will be `lldbsuite.lldb_root`
    lldbRootDirectory = lldbsuite.lldb_root

    # Some of the tests can invoke the 'lldb' command directly.
    # We'll try to locate the appropriate executable right here.

    # The lldb executable can be set from the command line
    # if it's not set, we try to find it now
    # first, we try the environment
    if not lldbtest_config.lldbExec:
        # First, you can define an environment variable LLDB_EXEC specifying the
        # full pathname of the lldb executable.
        if "LLDB_EXEC" in os.environ:
            lldbtest_config.lldbExec = os.environ["LLDB_EXEC"]

    if not lldbtest_config.lldbExec:
        outputPaths = getOutputPaths(lldbRootDirectory)
        for outputPath in outputPaths:
            candidatePath = os.path.join(outputPath, 'lldb')
            if is_exe(candidatePath):
                lldbtest_config.lldbExec = candidatePath
                break

    if not lldbtest_config.lldbExec:
        # Last, check the path
        lldbtest_config.lldbExec = which('lldb')

    if lldbtest_config.lldbExec and not is_exe(lldbtest_config.lldbExec):
        print("'{}' is not a path to a valid executable".format(lldbtest_config.lldbExec))
        lldbtest_config.lldbExec = None

    if not lldbtest_config.lldbExec:
        print("The 'lldb' executable cannot be located.  Some of the tests may not be run as a result.")
        sys.exit(-1)

    lldbLibDir = os.path.dirname(lldbtest_config.lldbExec)  # confusingly, this is the "bin" directory
    os.environ["LLDB_LIB_DIR"] = lldbLibDir
    lldbImpLibDir = os.path.join(lldbLibDir, '..', 'lib') if sys.platform.startswith('win32') else lldbLibDir
    os.environ["LLDB_IMPLIB_DIR"] = lldbImpLibDir
    if not noHeaders:
        print("LLDB library dir:", os.environ["LLDB_LIB_DIR"])
        print("LLDB import library dir:", os.environ["LLDB_IMPLIB_DIR"])
        os.system('%s -v' % lldbtest_config.lldbExec)

    # Assume lldb-mi is in same place as lldb
    # If not found, disable the lldb-mi tests
    lldbMiExec = None
    if lldbtest_config.lldbExec and is_exe(lldbtest_config.lldbExec + "-mi"):
        lldbMiExec = lldbtest_config.lldbExec + "-mi"
    if not lldbMiExec:
        if not shouldSkipBecauseOfCategories(["lldb-mi"]):
            print("The 'lldb-mi' executable cannot be located.  The lldb-mi tests can not be run as a result.")
            skipCategories.append("lldb-mi")
    else:
        os.environ["LLDBMI_EXEC"] = lldbMiExec

    # Skip printing svn/git information when running in parsable (lit-test compatibility) mode
    if not svn_silent and not parsable:
        if os.path.isdir(os.path.join(lldbRootDirectory, '.svn')) and which("svn") is not None:
            pipe = subprocess.Popen([which("svn"), "info", lldbRootDirectory], stdout = subprocess.PIPE)
            svn_info = pipe.stdout.read()
        elif os.path.isdir(os.path.join(lldbRootDirectory, '.git')) and which("git") is not None:
            pipe = subprocess.Popen([which("git"), "svn", "info", lldbRootDirectory], stdout = subprocess.PIPE)
            svn_info = pipe.stdout.read()
        if not noHeaders:
            print(svn_info)

    global ignore

    lldbPythonDir = None # The directory that contains 'lldb/__init__.py'
    if lldbFrameworkPath:
        candidatePath = os.path.join(lldbFrameworkPath, 'Resources', 'Python')
        if os.path.isfile(os.path.join(candidatePath, 'lldb/__init__.py')):
            lldbPythonDir = candidatePath
        if not lldbPythonDir:
            print('Resources/Python/lldb/__init__.py was not found in ' + lldbFrameworkPath)
            sys.exit(-1)
    else:
        # The '-i' option is used to skip looking for lldb.py in the build tree.
        if ignore:
            return
        
        # If our lldb supports the -P option, use it to find the python path:
        init_in_python_dir = os.path.join('lldb', '__init__.py')

        lldb_dash_p_result = subprocess.check_output([lldbtest_config.lldbExec, "-P"], stderr=subprocess.STDOUT, universal_newlines=True)

        if lldb_dash_p_result and not lldb_dash_p_result.startswith(("<", "lldb: invalid option:")) \
							  and not lldb_dash_p_result.startswith("Traceback"):
            lines = lldb_dash_p_result.splitlines()

            # Workaround for readline vs libedit issue on FreeBSD.  If stdout
            # is not a terminal Python executes
            #     rl_variable_bind ("enable-meta-key", "off");
            # This produces a warning with FreeBSD's libedit because the
            # enable-meta-key variable is unknown.  Not an issue on Apple
            # because cpython commit f0ab6f9f0603 added a #ifndef __APPLE__
            # around the call.  See http://bugs.python.org/issue19884 for more
            # information.  For now we just discard the warning output.
            if len(lines) >= 1 and lines[0].startswith("bind: Invalid command"):
                lines.pop(0)

            # Taking the last line because lldb outputs
            # 'Cannot read termcap database;\nusing dumb terminal settings.\n'
            # before the path
            if len(lines) >= 1 and os.path.isfile(os.path.join(lines[-1], init_in_python_dir)):
                lldbPythonDir = lines[-1]
                if "freebsd" in sys.platform or "linux" in sys.platform:
                    os.environ['LLDB_LIB_DIR'] = os.path.join(lldbPythonDir, '..', '..')
        
        if not lldbPythonDir:
            if platform.system() == "Darwin":
                python_resource_dir = ['LLDB.framework', 'Resources', 'Python']
                outputPaths = getXcodeOutputPaths()
                for outputPath in outputPaths:
                    candidatePath = os.path.join(outputPath, python_resource_dir)
                    if os.path.isfile(os.path.join(candidatePath, init_in_python_dir)):
                        lldbPythonDir = candidatePath
                        break

                if not lldbPythonDir:
                    print('This script requires lldb.py to be in either ' + dbgPath + ',', end=' ')
                    print(relPath + ', or ' + baiPath + '. Some tests might fail.')
            else:
                print("Unable to load lldb extension module.  Possible reasons for this include:")
                print("  1) LLDB was built with LLDB_DISABLE_PYTHON=1")
                print("  2) PYTHONPATH and PYTHONHOME are not set correctly.  PYTHONHOME should refer to")
                print("     the version of Python that LLDB built and linked against, and PYTHONPATH")
                print("     should contain the Lib directory for the same python distro, as well as the")
                print("     location of LLDB\'s site-packages folder.")
                print("  3) A different version of Python than that which was built against is exported in")
                print("     the system\'s PATH environment variable, causing conflicts.")
                print("  4) The executable '%s' could not be found.  Please check " % lldbExecutable)
                print("     that it exists and is executable.")

    if lldbPythonDir:
        lldbPythonDir = os.path.normpath(lldbPythonDir)
        # Some of the code that uses this path assumes it hasn't resolved the Versions... link.  
        # If the path we've constructed looks like that, then we'll strip out the Versions/A part.
        (before, frameWithVersion, after) = lldbPythonDir.rpartition("LLDB.framework/Versions/A")
        if frameWithVersion != "" :
            lldbPythonDir = before + "LLDB.framework" + after

        lldbPythonDir = os.path.abspath(lldbPythonDir)

        # If tests need to find LLDB_FRAMEWORK, now they can do it
        os.environ["LLDB_FRAMEWORK"] = os.path.dirname(os.path.dirname(lldbPythonDir))

        # This is to locate the lldb.py module.  Insert it right after sys.path[0].
        sys.path[1:1] = [lldbPythonDir]
        if dumpSysPath:
            print("sys.path:", sys.path)

def visit(prefix, dir, names):
    """Visitor function for os.path.walk(path, visit, arg)."""

    global suite
    global regexp
    global filters
    global fs4all
    global excluded
    global all_tests

    if set(dir.split(os.sep)).intersection(excluded):
        #print("Detected an excluded dir component: %s" % dir)
        return

    for name in names:
        if '.py' == os.path.splitext(name)[1] and name.startswith(prefix):

            if name in all_tests:
                raise Exception("Found multiple tests with the name %s" % name)
            all_tests.add(name)

            # Try to match the regexp pattern, if specified.
            if regexp:
                import re
                if re.search(regexp, name):
                    #print("Filename: '%s' matches pattern: '%s'" % (name, regexp))
                    pass
                else:
                    #print("Filename: '%s' does not match pattern: '%s'" % (name, regexp))
                    continue

            # We found a match for our test.  Add it to the suite.

            # Update the sys.path first.
            if not sys.path.count(dir):
                sys.path.insert(0, dir)
            base = os.path.splitext(name)[0]

            # Thoroughly check the filterspec against the base module and admit
            # the (base, filterspec) combination only when it makes sense.
            filterspec = None
            for filterspec in filters:
                # Optimistically set the flag to True.
                filtered = True
                module = __import__(base)
                parts = filterspec.split('.')
                obj = module
                for part in parts:
                    try:
                        parent, obj = obj, getattr(obj, part)
                    except AttributeError:
                        # The filterspec has failed.
                        filtered = False
                        break

                # If filtered, we have a good filterspec.  Add it.
                if filtered:
                    #print("adding filter spec %s to module %s" % (filterspec, module))
                    suite.addTests(
                        unittest2.defaultTestLoader.loadTestsFromName(filterspec, module))
                    continue

            # Forgo this module if the (base, filterspec) combo is invalid
            # and no '-g' option is specified
            if filters and fs4all and not filtered:
                continue

            # Add either the filtered test case(s) (which is done before) or the entire test class.
            if not filterspec or not filtered:
                # A simple case of just the module name.  Also the failover case
                # from the filterspec branch when the (base, filterspec) combo
                # doesn't make sense.
                suite.addTests(unittest2.defaultTestLoader.loadTestsFromName(base))


def disabledynamics():
    import lldb
    ci = lldb.DBG.GetCommandInterpreter()
    res = lldb.SBCommandReturnObject()
    ci.HandleCommand("setting set target.prefer-dynamic-value no-dynamic-values", res, False)    
    if not res.Succeeded():
        raise Exception('disabling dynamic type support failed')

def lldbLoggings():
    import lldb
    """Check and do lldb loggings if necessary."""

    # Turn on logging for debugging purposes if ${LLDB_LOG} environment variable is
    # defined.  Use ${LLDB_LOG} to specify the log file.
    ci = lldb.DBG.GetCommandInterpreter()
    res = lldb.SBCommandReturnObject()
    if ("LLDB_LOG" in os.environ):
        open(os.environ["LLDB_LOG"], 'w').close()
        if ("LLDB_LOG_OPTION" in os.environ):
            lldb_log_option = os.environ["LLDB_LOG_OPTION"]
        else:
            lldb_log_option = "event process expr state api"
        ci.HandleCommand(
            "log enable -n -f " + os.environ["LLDB_LOG"] + " lldb " + lldb_log_option,
            res)
        if not res.Succeeded():
            raise Exception('log enable failed (check LLDB_LOG env variable)')

    if ("LLDB_LINUX_LOG" in os.environ):
        open(os.environ["LLDB_LINUX_LOG"], 'w').close()
        if ("LLDB_LINUX_LOG_OPTION" in os.environ):
            lldb_log_option = os.environ["LLDB_LINUX_LOG_OPTION"]
        else:
            lldb_log_option = "event process expr state api"
        ci.HandleCommand(
            "log enable -n -f " + os.environ["LLDB_LINUX_LOG"] + " linux " + lldb_log_option,
            res)
        if not res.Succeeded():
            raise Exception('log enable failed (check LLDB_LINUX_LOG env variable)')
 
    # Ditto for gdb-remote logging if ${GDB_REMOTE_LOG} environment variable is defined.
    # Use ${GDB_REMOTE_LOG} to specify the log file.
    if ("GDB_REMOTE_LOG" in os.environ):
        if ("GDB_REMOTE_LOG_OPTION" in os.environ):
            gdb_remote_log_option = os.environ["GDB_REMOTE_LOG_OPTION"]
        else:
            gdb_remote_log_option = "packets process"
        ci.HandleCommand(
            "log enable -n -f " + os.environ["GDB_REMOTE_LOG"] + " gdb-remote "
            + gdb_remote_log_option,
            res)
        if not res.Succeeded():
            raise Exception('log enable failed (check GDB_REMOTE_LOG env variable)')

def getMyCommandLine():
    return ' '.join(sys.argv)

# ======================================== #
#                                          #
# Execution of the test driver starts here #
#                                          #
# ======================================== #

def checkDsymForUUIDIsNotOn():
    cmd = ["defaults", "read", "com.apple.DebugSymbols"]
    pipe = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    cmd_output = pipe.stdout.read()
    if cmd_output and "DBGFileMappedPaths = " in cmd_output:
        print("%s =>" % ' '.join(cmd))
        print(cmd_output)
        print("Disable automatic lookup and caching of dSYMs before running the test suite!")
        print("Exiting...")
        sys.exit(0)

def exitTestSuite(exitCode = None):
    import lldb
    lldb.SBDebugger.Terminate()
    if exitCode:
        sys.exit(exitCode)


def isMultiprocessTestRunner():
    # We're not multiprocess when we're either explicitly
    # the inferior (as specified by the multiprocess test
    # runner) OR we've been told to skip using the multiprocess
    # test runner
    return not (is_inferior_test_runner or no_multiprocess_test_runner)

def getVersionForSDK(sdk):
    sdk = str.lower(sdk)
    full_path = seven.get_command_output('xcrun -sdk %s --show-sdk-path' % sdk)
    basename = os.path.basename(full_path)
    basename = os.path.splitext(basename)[0]
    basename = str.lower(basename)
    ver = basename.replace(sdk, '')
    return ver

def getPathForSDK(sdk):
    sdk = str.lower(sdk)
    full_path = seven.get_command_output('xcrun -sdk %s --show-sdk-path' % sdk)
    if os.path.exists(full_path): return full_path
    return None

def setDefaultTripleForPlatform():
    if lldb_platform_name == 'ios-simulator':
        triple_str = 'x86_64-apple-ios%s' % (getVersionForSDK('iphonesimulator'))
        os.environ['TRIPLE'] = triple_str
        return {'TRIPLE':triple_str}
    return {}

def run_suite():
    global just_do_benchmarks_test
    global dont_do_dsym_test
    global dont_do_dwarf_test
    global dont_do_dwo_test
    global blacklist
    global blacklistConfig
    global categoriesList
    global validCategories
    global useCategories
    global skipCategories
    global lldbFrameworkPath
    global configFile
    global archs
    global compilers
    global count
    global dumpSysPath
    global bmExecutable
    global bmBreakpointSpec
    global bmIterationCount
    global failed
    global failfast
    global filters
    global fs4all
    global ignore
    global progress_bar
    global runHooks
    global skip_build_and_cleanup
    global skip_long_running_test
    global noHeaders
    global parsable
    global regexp
    global rdir
    global sdir_name
    global svn_silent
    global verbose
    global testdirs
    global lldb_platform_name
    global lldb_platform_url
    global lldb_platform_working_dir
    global setCrashInfoHook
    global is_inferior_test_runner
    global multiprocess_test_subdir
    global num_threads
    global output_on_success
    global no_multiprocess_test_runner
    global test_runner_name
    global results_filename
    global results_formatter_name
    global results_formatter_options
    global results_port

    # On MacOS X, check to make sure that domain for com.apple.DebugSymbols defaults
    # does not exist before proceeding to running the test suite.
    if sys.platform.startswith("darwin"):
        checkDsymForUUIDIsNotOn()

    #
    # Start the actions by first parsing the options while setting up the test
    # directories, followed by setting up the search paths for lldb utilities;
    # then, we walk the directory trees and collect the tests into our test suite.
    #
    parseOptionsAndInitTestdirs()

    # Setup test results (test results formatter and output handling).
    setupTestResults()

    # If we are running as the multiprocess test runner, kick off the
    # multiprocess test runner here.
    if isMultiprocessTestRunner():
        from . import dosep
        dosep.main(output_on_success, num_threads, multiprocess_test_subdir,
                   test_runner_name, results_formatter_object)
        raise Exception("should never get here")
    elif is_inferior_test_runner:
        # Shut off Ctrl-C processing in inferiors.  The parallel
        # test runner handles this more holistically.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    setupSysPath()
    setupCrashInfoHook()

    #
    # If '-l' is specified, do not skip the long running tests.
    if not skip_long_running_test:
        os.environ["LLDB_SKIP_LONG_RUNNING_TEST"] = "NO"

    # For the time being, let's bracket the test runner within the
    # lldb.SBDebugger.Initialize()/Terminate() pair.
    import lldb

    # Create a singleton SBDebugger in the lldb namespace.
    lldb.DBG = lldb.SBDebugger.Create()

    if lldb_platform_name:
        print("Setting up remote platform '%s'" % (lldb_platform_name))
        lldb.remote_platform = lldb.SBPlatform(lldb_platform_name)
        lldb.remote_platform_name = lldb_platform_name
        if not lldb.remote_platform.IsValid():
            print("error: unable to create the LLDB platform named '%s'." % (lldb_platform_name))
            exitTestSuite(1)
        if lldb_platform_url:
            # We must connect to a remote platform if a LLDB platform URL was specified
            print("Connecting to remote platform '%s' at '%s'..." % (lldb_platform_name, lldb_platform_url))
            lldb.platform_url = lldb_platform_url
            platform_connect_options = lldb.SBPlatformConnectOptions(lldb_platform_url)
            err = lldb.remote_platform.ConnectRemote(platform_connect_options)
            if err.Success():
                print("Connected.")
            else:
                print("error: failed to connect to remote platform using URL '%s': %s" % (lldb_platform_url, err))
                exitTestSuite(1)
        else:
            lldb.platform_url = None

    platform_changes = setDefaultTripleForPlatform()
    first = True
    for key in platform_changes:
        if first:
            print("Environment variables setup for platform support:")
            first = False
        print("%s = %s" % (key,platform_changes[key]))

    if lldb_platform_working_dir:
        print("Setting remote platform working directory to '%s'..." % (lldb_platform_working_dir))
        lldb.remote_platform.SetWorkingDirectory(lldb_platform_working_dir)
        lldb.remote_platform_working_dir = lldb_platform_working_dir
        lldb.DBG.SetSelectedPlatform(lldb.remote_platform)
    else:
        lldb.remote_platform = None
        lldb.remote_platform_working_dir = None
        lldb.platform_url = None

    target_platform = lldb.DBG.GetSelectedPlatform().GetTriple().split('-')[2]

    # By default, both dsym, dwarf and dwo tests are performed.
    # Use @dsym_test, @dwarf_test or @dwo_test decorators, defined in lldbtest.py, to mark a test as
    # a dsym, dwarf or dwo test.  Use '-N dsym', '-N dwarf' or '-N dwo' to exclude dsym, dwarf or
    # dwo tests from running.
    dont_do_dsym_test = dont_do_dsym_test or any(platform in target_platform for platform in ["linux", "freebsd", "windows"])
    dont_do_dwo_test = dont_do_dwo_test or any(platform in target_platform for platform in ["darwin", "macosx", "ios"])

    # Don't do debugserver tests on everything except OS X.
    dont_do_debugserver_test = "linux" in target_platform or "freebsd" in target_platform or "windows" in target_platform

    # Don't do lldb-server (llgs) tests on anything except Linux.
    dont_do_llgs_test = not ("linux" in target_platform)

    #
    # Walk through the testdirs while collecting tests.
    #
    for testdir in testdirs:
        for (dirpath, dirnames, filenames) in os.walk(testdir):
            visit('Test', dirpath, filenames)

    #
    # Now that we have loaded all the test cases, run the whole test suite.
    #

    # Put the blacklist in the lldb namespace, to be used by lldb.TestBase.
    lldb.blacklist = blacklist

    # The pre_flight and post_flight come from reading a config file.
    lldb.pre_flight = pre_flight
    lldb.post_flight = post_flight
    def getsource_if_available(obj):
        """
        Return the text of the source code for an object if available.  Otherwise,
        a print representation is returned.
        """
        import inspect
        try:
            return inspect.getsource(obj)
        except:
            return repr(obj)

    if not noHeaders:
        print("lldb.pre_flight:", getsource_if_available(lldb.pre_flight))
        print("lldb.post_flight:", getsource_if_available(lldb.post_flight))

    # If either pre_flight or post_flight is defined, set lldb.test_remote to True.
    if lldb.pre_flight or lldb.post_flight:
        lldb.test_remote = True
    else:
        lldb.test_remote = False

    # So do the lldbtest_remote_sandbox and lldbtest_remote_shell_template variables.
    lldb.lldbtest_remote_sandbox = lldbtest_remote_sandbox
    lldb.lldbtest_remote_sandboxed_executable = None
    lldb.lldbtest_remote_shell_template = lldbtest_remote_shell_template

    # Put all these test decorators in the lldb namespace.
    lldb.just_do_benchmarks_test = just_do_benchmarks_test
    lldb.dont_do_dsym_test = dont_do_dsym_test
    lldb.dont_do_dwarf_test = dont_do_dwarf_test
    lldb.dont_do_dwo_test = dont_do_dwo_test
    lldb.dont_do_debugserver_test = dont_do_debugserver_test
    lldb.dont_do_llgs_test = dont_do_llgs_test

    # Do we need to skip build and cleanup?
    lldb.skip_build_and_cleanup = skip_build_and_cleanup

    # Put bmExecutable, bmBreakpointSpec, and bmIterationCount into the lldb namespace, too.
    lldb.bmExecutable = bmExecutable
    lldb.bmBreakpointSpec = bmBreakpointSpec
    lldb.bmIterationCount = bmIterationCount

    # And don't forget the runHooks!
    lldb.runHooks = runHooks

    # Turn on lldb loggings if necessary.
    lldbLoggings()

    # Disable default dynamic types for testing purposes
    disabledynamics()

    # Install the control-c handler.
    unittest2.signals.installHandler()

    # If sdir_name is not specified through the '-s sdir_name' option, get a
    # timestamp string and export it as LLDB_SESSION_DIR environment var.  This will
    # be used when/if we want to dump the session info of individual test cases
    # later on.
    #
    # See also TestBase.dumpSessionInfo() in lldbtest.py.
    import datetime
    # The windows platforms don't like ':' in the pathname.
    timestamp_started = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    if not sdir_name:
        sdir_name = timestamp_started
    os.environ["LLDB_SESSION_DIRNAME"] = os.path.join(os.getcwd(), sdir_name)

    if not noHeaders:
        sys.stderr.write("\nSession logs for test failures/errors/unexpected successes"
                         " will go into directory '%s'\n" % sdir_name)
        sys.stderr.write("Command invoked: %s\n" % getMyCommandLine())

    if not os.path.isdir(sdir_name):
        try:
            os.mkdir(sdir_name)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    where_to_save_session = os.getcwd()
    fname = os.path.join(sdir_name, "TestStarted-%d" % os.getpid())
    with open(fname, "w") as f:
        print("Test started at: %s\n" % timestamp_started, file=f)
        print(svn_info, file=f)
        print("Command invoked: %s\n" % getMyCommandLine(), file=f)

    #
    # Invoke the default TextTestRunner to run the test suite, possibly iterating
    # over different configurations.
    #

    iterArchs = False
    iterCompilers = False

    if not archs and "archs" in config:
        archs = config["archs"]

    if isinstance(archs, list) and len(archs) >= 1:
        iterArchs = True

    if not compilers and "compilers" in config:
        compilers = config["compilers"]

    #
    # Add some intervention here to sanity check that the compilers requested are sane.
    # If found not to be an executable program, the invalid one is dropped from the list.
    for i in range(len(compilers)):
        c = compilers[i]
        if which(c):
            continue
        else:
            if sys.platform.startswith("darwin"):
                pipe = subprocess.Popen(['xcrun', '-find', c], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
                cmd_output = pipe.stdout.read()
                if cmd_output:
                    if "not found" in cmd_output:
                        print("dropping %s from the compilers used" % c)
                        compilers.remove(i)
                    else:
                        compilers[i] = cmd_output.split('\n')[0]
                        print("'xcrun -find %s' returning %s" % (c, compilers[i]))

    if not parsable:
        print("compilers=%s" % str(compilers))

    if not compilers or len(compilers) == 0:
        print("No eligible compiler found, exiting.")
        exitTestSuite(1)

    if isinstance(compilers, list) and len(compilers) >= 1:
        iterCompilers = True

    # Make a shallow copy of sys.path, we need to manipulate the search paths later.
    # This is only necessary if we are relocated and with different configurations.
    if rdir:
        old_sys_path = sys.path[:]
    # If we iterate on archs or compilers, there is a chance we want to split stderr/stdout.
    if iterArchs or iterCompilers:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        new_stderr = None
        new_stdout = None

    # Iterating over all possible architecture and compiler combinations.
    for ia in range(len(archs) if iterArchs else 1):
        archConfig = ""
        if iterArchs:
            os.environ["ARCH"] = archs[ia]
            archConfig = "arch=%s" % archs[ia]
        for ic in range(len(compilers) if iterCompilers else 1):
            if iterCompilers:
                os.environ["CC"] = compilers[ic]
                configString = "%s compiler=%s" % (archConfig, compilers[ic])
            else:
                configString = archConfig

            if iterArchs or iterCompilers:
                # Translate ' ' to '-' for pathname component.
                if six.PY2:
                    import string
                    tbl = string.maketrans(' ', '-')
                else:
                    tbl = str.maketrans(' ', '-')
                configPostfix = configString.translate(tbl)

                # Check whether we need to split stderr/stdout into configuration
                # specific files.
                if old_stderr.name != '<stderr>' and config.get('split_stderr'):
                    if new_stderr:
                        new_stderr.close()
                    new_stderr = open("%s.%s" % (old_stderr.name, configPostfix), "w")
                    sys.stderr = new_stderr
                if old_stdout.name != '<stdout>' and config.get('split_stdout'):
                    if new_stdout:
                        new_stdout.close()
                    new_stdout = open("%s.%s" % (old_stdout.name, configPostfix), "w")
                    sys.stdout = new_stdout

                # If we specified a relocated directory to run the test suite, do
                # the extra housekeeping to copy the testdirs to a configStringified
                # directory and to update sys.path before invoking the test runner.
                # The purpose is to separate the configuration-specific directories
                # from each other.
                if rdir:
                    from shutil import copytree, rmtree, ignore_patterns

                    newrdir = "%s.%s" % (rdir, configPostfix)

                    # Copy the tree to a new directory with postfix name configPostfix.
                    if os.path.exists(newrdir):
                        rmtree(newrdir)
                    copytree(rdir, newrdir, ignore=ignore_patterns('*.pyc', '*.o', '*.d'))

                    # Update the LLDB_TEST environment variable to reflect new top
                    # level test directory.
                    #
                    # See also lldbtest.TestBase.setUpClass(cls).
                    if len(testdirs) == 1 and os.path.basename(testdirs[0]) == 'test':
                        os.environ["LLDB_TEST"] = os.path.join(newrdir, 'test')
                    else:
                        os.environ["LLDB_TEST"] = newrdir

                    # And update the Python search paths for modules.
                    sys.path = [x.replace(rdir, newrdir, 1) for x in old_sys_path]

                # Output the configuration.
                if not parsable:
                    sys.stderr.write("\nConfiguration: " + configString + "\n")

            #print("sys.stderr name is", sys.stderr.name)
            #print("sys.stdout name is", sys.stdout.name)

            # First, write out the number of collected test cases.
            if not parsable:
                sys.stderr.write(separator + "\n")
                sys.stderr.write("Collected %d test%s\n\n"
                                 % (suite.countTestCases(),
                                    suite.countTestCases() != 1 and "s" or ""))

            class LLDBTestResult(unittest2.TextTestResult):
                """
                Enforce a singleton pattern to allow introspection of test progress.

                Overwrite addError(), addFailure(), and addExpectedFailure() methods
                to enable each test instance to track its failure/error status.  It
                is used in the LLDB test framework to emit detailed trace messages
                to a log file for easier human inspection of test failures/errors.
                """
                __singleton__ = None
                __ignore_singleton__ = False

                @staticmethod
                def getTerminalSize():
                    import os
                    env = os.environ
                    def ioctl_GWINSZ(fd):
                        try:
                            import fcntl, termios, struct, os
                            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
                        '1234'))
                        except:
                            return
                        return cr
                    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
                    if not cr:
                        try:
                            fd = os.open(os.ctermid(), os.O_RDONLY)
                            cr = ioctl_GWINSZ(fd)
                            os.close(fd)
                        except:
                            pass
                    if not cr:
                        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
                    return int(cr[1]), int(cr[0])

                def __init__(self, *args):
                    if not LLDBTestResult.__ignore_singleton__ and LLDBTestResult.__singleton__:
                        raise Exception("LLDBTestResult instantiated more than once")
                    super(LLDBTestResult, self).__init__(*args)
                    LLDBTestResult.__singleton__ = self
                    # Now put this singleton into the lldb module namespace.
                    lldb.test_result = self
                    # Computes the format string for displaying the counter.
                    global suite
                    counterWidth = len(str(suite.countTestCases()))
                    self.fmt = "%" + str(counterWidth) + "d: "
                    self.indentation = ' ' * (counterWidth + 2)
                    # This counts from 1 .. suite.countTestCases().
                    self.counter = 0
                    (width, height) = LLDBTestResult.getTerminalSize()
                    self.progressbar = None
                    global progress_bar
                    if width > 10 and not parsable and progress_bar:
                        try:
                            self.progressbar = progress.ProgressWithEvents(stdout=self.stream,start=0,end=suite.countTestCases(),width=width-10)
                        except:
                            self.progressbar = None
                    self.results_formatter = results_formatter_object

                def _config_string(self, test):
                  compiler = getattr(test, "getCompiler", None)
                  arch = getattr(test, "getArchitecture", None)
                  return "%s-%s" % (compiler() if compiler else "", arch() if arch else "")

                def _exc_info_to_string(self, err, test):
                    """Overrides superclass TestResult's method in order to append
                    our test config info string to the exception info string."""
                    if hasattr(test, "getArchitecture") and hasattr(test, "getCompiler"):
                        return '%sConfig=%s-%s' % (super(LLDBTestResult, self)._exc_info_to_string(err, test),
                                                                  test.getArchitecture(),
                                                                  test.getCompiler())
                    else:
                        return super(LLDBTestResult, self)._exc_info_to_string(err, test)

                def getDescription(self, test):
                    doc_first_line = test.shortDescription()
                    if self.descriptions and doc_first_line:
                        return '\n'.join((str(test), self.indentation + doc_first_line))
                    else:
                        return str(test)

                def getCategoriesForTest(self,test):
                    if hasattr(test,"_testMethodName"):
                        test_method = getattr(test,"_testMethodName")
                        test_method = getattr(test,test_method)
                    else:
                        test_method = None
                    if test_method != None and hasattr(test_method,"getCategories"):
                        test_categories = test_method.getCategories(test)
                    elif hasattr(test,"getCategories"):
                        test_categories = test.getCategories()
                    elif inspect.ismethod(test) and test.__self__ != None and hasattr(test.__self__,"getCategories"):
                        test_categories = test.__self__.getCategories()
                    else:
                        test_categories = []
                    if test_categories == None:
                        test_categories = []
                    return test_categories

                def hardMarkAsSkipped(self,test):
                    getattr(test, test._testMethodName).__func__.__unittest_skip__ = True
                    getattr(test, test._testMethodName).__func__.__unittest_skip_why__ = "test case does not fall in any category of interest for this run"
                    test.__class__.__unittest_skip__ = True
                    test.__class__.__unittest_skip_why__ = "test case does not fall in any category of interest for this run"

                def startTest(self, test):
                    if shouldSkipBecauseOfCategories(self.getCategoriesForTest(test)):
                        self.hardMarkAsSkipped(test)
                    global setCrashInfoHook
                    setCrashInfoHook("%s at %s" % (str(test),inspect.getfile(test.__class__)))
                    self.counter += 1
                    #if self.counter == 4:
                    #    import crashinfo
                    #    crashinfo.testCrashReporterDescription(None)
                    test.test_number = self.counter
                    if self.showAll:
                        self.stream.write(self.fmt % self.counter)
                    super(LLDBTestResult, self).startTest(test)
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_start(test))

                def addSuccess(self, test):
                    global parsable
                    super(LLDBTestResult, self).addSuccess(test)
                    if parsable:
                        self.stream.write("PASS: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_success(test))

                def addError(self, test, err):
                    global sdir_has_content
                    global parsable
                    sdir_has_content = True
                    super(LLDBTestResult, self).addError(test, err)
                    method = getattr(test, "markError", None)
                    if method:
                        method()
                    if parsable:
                        self.stream.write("FAIL: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_error(test, err))

                def addCleanupError(self, test, err):
                    global sdir_has_content
                    global parsable
                    sdir_has_content = True
                    super(LLDBTestResult, self).addCleanupError(test, err)
                    method = getattr(test, "markCleanupError", None)
                    if method:
                        method()
                    if parsable:
                        self.stream.write("CLEANUP ERROR: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_cleanup_error(
                                test, err))

                def addFailure(self, test, err):
                    global sdir_has_content
                    global failuresPerCategory
                    global parsable
                    sdir_has_content = True
                    super(LLDBTestResult, self).addFailure(test, err)
                    method = getattr(test, "markFailure", None)
                    if method:
                        method()
                    if parsable:
                        self.stream.write("FAIL: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
                    if useCategories:
                        test_categories = self.getCategoriesForTest(test)
                        for category in test_categories:
                            if category in failuresPerCategory:
                                failuresPerCategory[category] = failuresPerCategory[category] + 1
                            else:
                                failuresPerCategory[category] = 1
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_failure(test, err))


                def addExpectedFailure(self, test, err, bugnumber):
                    global sdir_has_content
                    global parsable
                    sdir_has_content = True
                    super(LLDBTestResult, self).addExpectedFailure(test, err, bugnumber)
                    method = getattr(test, "markExpectedFailure", None)
                    if method:
                        method(err, bugnumber)
                    if parsable:
                        self.stream.write("XFAIL: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_expected_failure(
                            test, err, bugnumber))

                def addSkip(self, test, reason):
                    global sdir_has_content
                    global parsable
                    sdir_has_content = True
                    super(LLDBTestResult, self).addSkip(test, reason)
                    method = getattr(test, "markSkippedTest", None)
                    if method:
                        method()
                    if parsable:
                        self.stream.write("UNSUPPORTED: LLDB (%s) :: %s (%s) \n" % (self._config_string(test), str(test), reason))
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_skip(test, reason))

                def addUnexpectedSuccess(self, test, bugnumber):
                    global sdir_has_content
                    global parsable
                    sdir_has_content = True
                    super(LLDBTestResult, self).addUnexpectedSuccess(test, bugnumber)
                    method = getattr(test, "markUnexpectedSuccess", None)
                    if method:
                        method(bugnumber)
                    if parsable:
                        self.stream.write("XPASS: LLDB (%s) :: %s\n" % (self._config_string(test), str(test)))
                    if self.results_formatter:
                        self.results_formatter.handle_event(
                            EventBuilder.event_for_unexpected_success(
                                test, bugnumber))


            if parsable:
                v = 0
            elif progress_bar:
                v = 1
            else:
                v = verbose

            # Invoke the test runner.
            if count == 1:
                result = unittest2.TextTestRunner(stream=sys.stderr,
                                                  verbosity=v,
                                                  failfast=failfast,
                                                  resultclass=LLDBTestResult).run(suite)
            else:
                # We are invoking the same test suite more than once.  In this case,
                # mark __ignore_singleton__ flag as True so the signleton pattern is
                # not enforced.
                LLDBTestResult.__ignore_singleton__ = True
                for i in range(count):
               
                    result = unittest2.TextTestRunner(stream=sys.stderr,
                                                      verbosity=v,
                                                      failfast=failfast,
                                                      resultclass=LLDBTestResult).run(suite)

            failed = failed or not result.wasSuccessful()

    if sdir_has_content and not parsable:
        sys.stderr.write("Session logs for test failures/errors/unexpected successes"
                         " can be found in directory '%s'\n" % sdir_name)

    if useCategories and len(failuresPerCategory) > 0:
        sys.stderr.write("Failures per category:\n")
        for category in failuresPerCategory:
            sys.stderr.write("%s - %d\n" % (category,failuresPerCategory[category]))

    os.chdir(where_to_save_session)
    fname = os.path.join(sdir_name, "TestFinished-%d" % os.getpid())
    with open(fname, "w") as f:
        print("Test finished at: %s\n" % datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"), file=f)

    # Terminate the test suite if ${LLDB_TESTSUITE_FORCE_FINISH} is defined.
    # This should not be necessary now.
    if ("LLDB_TESTSUITE_FORCE_FINISH" in os.environ):
        print("Terminating Test suite...")
        subprocess.Popen(["/bin/sh", "-c", "kill %s; exit 0" % (os.getpid())])

    # Exiting.
    exitTestSuite(failed)

if __name__ == "__main__":
    print(__file__ + " is for use as a module only.  It should not be run as a standalone script.")
    sys.exit(-1)
