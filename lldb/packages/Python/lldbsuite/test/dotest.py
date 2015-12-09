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
from . import configuration
from . import dotest_args
from . import lldbtest_config
from . import test_categories
from . import result_formatter
from . import test_result
from .result_formatter import EventBuilder
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
def usage(parser):
    parser.print_help()
    if configuration.verbose > 0:
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

def parseOptionsAndInitTestdirs():
    """Initialize the list of directories containing our unittest scripts.

    '-h/--help as the first option prints out usage info and exit the program.
    """

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
        configuration.compilers = args.compilers
    else:
        # Use a compiler appropriate appropriate for the Apple SDK if one was specified
        if platform_system == 'Darwin' and args.apple_sdk:
            configuration.compilers = [seven.get_command_output('xcrun -sdk "%s" -find clang 2> /dev/null' % (args.apple_sdk))]
        else:
            # 'clang' on ubuntu 14.04 is 3.4 so we try clang-3.5 first
            candidateCompilers = ['clang-3.5', 'clang', 'gcc']
            for candidate in candidateCompilers:
                if which(candidate):
                    configuration.compilers = [candidate]
                    break

    if args.channels:
        lldbtest_config.channels = args.channels

    if args.log_success:
        lldbtest_config.log_success = args.log_success

    # Set SDKROOT if we are using an Apple SDK
    if platform_system == 'Darwin' and args.apple_sdk:
        os.environ['SDKROOT'] = seven.get_command_output('xcrun --sdk "%s" --show-sdk-path 2> /dev/null' % (args.apple_sdk))

    if args.archs:
        configuration.archs = args.archs
        for arch in configuration.archs:
            if arch.startswith('arm') and platform_system == 'Darwin' and not args.apple_sdk:
                os.environ['SDKROOT'] = seven.get_command_output('xcrun --sdk iphoneos.internal --show-sdk-path 2> /dev/null')
                if not os.path.exists(os.environ['SDKROOT']):
                    os.environ['SDKROOT'] = seven.get_command_output('xcrun --sdk iphoneos --show-sdk-path 2> /dev/null')
    else:
        configuration.archs = [platform_machine]

    if args.categoriesList:
        configuration.categoriesList = set(test_categories.validate(args.categoriesList, False))
        configuration.useCategories = True
    else:
        configuration.categoriesList = []

    if args.skipCategories:
        configuration.skipCategories = test_categories.validate(args.skipCategories, False)

    if args.E:
        cflags_extras = args.E
        os.environ['CFLAGS_EXTRAS'] = cflags_extras

    # argparse makes sure we have correct options
    if args.N == 'dwarf':
        configuration.dont_do_dwarf_test = True
    elif args.N == 'dwo':
        configuration.dont_do_dwo_test = True
    elif args.N == 'dsym':
        configuration.dont_do_dsym_test = True

    if args.a or args.plus_a:
        print("Options '-a' and '+a' have been deprecated. Please use the test category\n"
              "functionality (-G pyapi, --skip-category pyapi) instead.")
        sys.exit(1)

    if args.m or args.plus_m:
        print("Options '-m' and '+m' have been deprecated. Please use the test category\n"
              "functionality (-G lldb-mi, --skip-category lldb-mi) instead.")
        sys.exit(1)

    if args.d:
        sys.stdout.write("Suspending the process %d to wait for debugger to attach...\n" % os.getpid())
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGSTOP)

    if args.e:
        if args.e.startswith('-'):
            usage(parser)
        configuration.bmExecutable = args.e
        if not is_exe(configuration.bmExecutable):
            usage(parser)

    if args.F:
        configuration.failfast = True

    if args.f:
        if any([x.startswith('-') for x in args.f]):
            usage(parser)
        configuration.filters.extend(args.f)
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
        configuration.no_multiprocess_test_runner = True

    if args.i:
        configuration.ignore = True

    if args.k:
        configuration.runHooks.extend(args.k)

    if args.l:
        configuration.skip_long_running_test = False

    if args.framework:
        configuration.lldbFrameworkPath = args.framework

    if args.executable:
        lldbtest_config.lldbExec = args.executable

    if args.libcxx:
        os.environ["LIBCXX_PATH"] = args.libcxx

    if args.n:
        configuration.noHeaders = True

    if args.p:
        if args.p.startswith('-'):
            usage(parser)
        configuration.regexp = args.p

    if args.q:
        configuration.noHeaders = True
        configuration.parsable = True

    if args.P and not args.v:
        configuration.progress_bar = True
        configuration.verbose = 0

    if args.R:
        if args.R.startswith('-'):
            usage(parser)
        configuration.rdir = os.path.abspath(args.R)
        if os.path.exists(configuration.rdir):
            import shutil
            print('Removing tree:', configuration.rdir)
            shutil.rmtree(configuration.rdir)

    if args.r:
        if args.r.startswith('-'):
            usage(parser)
        configuration.rdir = os.path.abspath(args.r)
        if os.path.exists(configuration.rdir):
            print('Relocated directory:', configuration.rdir, 'must not exist!')
            usage(parser)

    if args.S:
        configuration.skip_build_and_cleanup = True

    if args.s:
        if args.s.startswith('-'):
            usage(parser)
        configuration.sdir_name = args.s

    if args.t:
        os.environ['LLDB_COMMAND_TRACE'] = 'YES'

    if args.T:
        configuration.svn_silent = False

    if args.v:
        configuration.verbose = 2

    if args.w:
        os.environ['LLDB_WAIT_BETWEEN_TEST_CASES'] = 'YES'

    if args.x:
        if args.x.startswith('-'):
            usage(parser)
        configuration.bmBreakpointSpec = args.x

    # argparse makes sure we have a number
    if args.y:
        configuration.bmIterationCount = args.y

    # argparse makes sure we have a number
    if args.sharp:
        configuration.count = args.sharp

    if sys.platform.startswith('win32'):
        os.environ['LLDB_DISABLE_CRASH_DIALOG'] = str(args.disable_crash_dialog)
        os.environ['LLDB_LAUNCH_INFERIORS_WITHOUT_CONSOLE'] = str(args.hide_inferior_console)

    if do_help == True:
        usage(parser)

    if args.no_multiprocess:
        configuration.no_multiprocess_test_runner = True

    if args.inferior:
        configuration.is_inferior_test_runner = True

    # Turn on output_on_sucess if either explicitly added or -v specified.
    if args.output_on_success or args.v:
        configuration.output_on_success = True

    if args.num_threads:
        configuration.num_threads = args.num_threads

    if args.test_subdir:
        configuration.multiprocess_test_subdir = args.test_subdir

    if args.test_runner_name:
        configuration.test_runner_name = args.test_runner_name

    # Capture test results-related args.
    if args.curses and not args.inferior:
        # Act as if the following args were set.
        args.results_formatter = "lldbsuite.test.curses_results.Curses"
        args.results_file = "stdout"

    if args.results_file:
        configuration.results_filename = args.results_file

    if args.results_port:
        configuration.results_port = args.results_port

    if args.results_file and args.results_port:
        sys.stderr.write(
            "only one of --results-file and --results-port should "
            "be specified\n")
        usage(args)

    if args.results_formatter:
        configuration.results_formatter_name = args.results_formatter
    if args.results_formatter_options:
        configuration.results_formatter_options = args.results_formatter_options

    if args.lldb_platform_name:
        configuration.lldb_platform_name = args.lldb_platform_name
    if args.lldb_platform_url:
        configuration.lldb_platform_url = args.lldb_platform_url
    if args.lldb_platform_working_dir:
        configuration.lldb_platform_working_dir = args.lldb_platform_working_dir

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
            result_formatter.EventBuilder.add_entries_to_all_events(entries)

    # Gather all the dirs passed on the command line.
    if len(args.args) > 0:
        configuration.testdirs = list(map(os.path.abspath, args.args))
        # Shut off multiprocessing mode when test directories are specified.
        configuration.no_multiprocess_test_runner = True

    # If '-r dir' is specified, the tests should be run under the relocated
    # directory.  Let's copy the testdirs over.
    if configuration.rdir:
        from shutil import copytree, ignore_patterns

        tmpdirs = []
        orig_testdirs = configuration.testdirs[:]
        for srcdir in configuration.testdirs:
            # For example, /Volumes/data/lldb/svn/ToT/test/functionalities/watchpoint/hello_watchpoint
            # shall be split into ['/Volumes/data/lldb/svn/ToT/', 'functionalities/watchpoint/hello_watchpoint'].
            # Utilize the relative path to the 'test' directory to make our destination dir path.
            if ("test" + os.sep) in srcdir:
                to_split_on = "test" + os.sep
            else:
                to_split_on = "test"
            dstdir = os.path.join(configuration.rdir, srcdir.split(to_split_on)[1])
            dstdir = dstdir.rstrip(os.sep)
            # Don't copy the *.pyc and .svn stuffs.
            copytree(srcdir, dstdir, ignore=ignore_patterns('*.pyc', '.svn'))
            tmpdirs.append(dstdir)

        # This will be our modified testdirs.
        configuration.testdirs = tmpdirs

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
        if len(configuration.testdirs) != 1 or os.path.basename(orig_testdirs[0]) != 'test':
            scriptdir = os.path.dirname(__file__)
            # Don't copy the .svn stuffs.
            copytree(os.path.join(scriptdir, 'make'), os.path.join(rdir, 'make'),
                     ignore=ignore_patterns('.svn'))

    #print("testdirs:", testdirs)

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

    # Create the results formatter.
    formatter_spec = result_formatter.create_results_formatter()
    if formatter_spec is not None and formatter_spec.formatter is not None:
        configuration.results_formatter_object = formatter_spec.formatter

        # Send an intialize message to the formatter.
        initialize_event = EventBuilder.bare_event("initialize")
        if isMultiprocessTestRunner():
            if configuration.test_runner_name is not None and configuration.test_runner_name == "serial":
                # Only one worker queue here.
                worker_count = 1
            else:
                # Workers will be the number of threads specified.
                worker_count = configuration.num_threads
        else:
            worker_count = 1
        initialize_event["worker_count"] = worker_count

        formatter_spec.formatter.handle_event(initialize_event)

        # Make sure we clean up the formatter on shutdown.
        if formatter_spec.cleanup_func is not None:
            atexit.register(formatter_spec.cleanup_func)


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

    # Get the directory containing the current script.
    if "DOTEST_PROFILE" in os.environ and "DOTEST_SCRIPT_DIR" in os.environ:
        scriptPath = os.environ["DOTEST_SCRIPT_DIR"]
    else:
        scriptPath = os.path.dirname(os.path.realpath(__file__))
    if not scriptPath.endswith('test'):
        print("This script expects to reside in lldb's test directory.")
        sys.exit(-1)

    if configuration.rdir:
        # Set up the LLDB_TEST environment variable appropriately, so that the
        # individual tests can be located relatively.
        #
        # See also lldbtest.TestBase.setUpClass(cls).
        if len(configuration.testdirs) == 1 and os.path.basename(configuration.testdirs[0]) == 'test':
            os.environ["LLDB_TEST"] = os.path.join(configuration.rdir, 'test')
        else:
            os.environ["LLDB_TEST"] = configuration.rdir
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
    if not configuration.noHeaders:
        print("LLDB library dir:", os.environ["LLDB_LIB_DIR"])
        print("LLDB import library dir:", os.environ["LLDB_IMPLIB_DIR"])
        os.system('%s -v' % lldbtest_config.lldbExec)

    # Assume lldb-mi is in same place as lldb
    # If not found, disable the lldb-mi tests
    lldbMiExec = None
    if lldbtest_config.lldbExec and is_exe(lldbtest_config.lldbExec + "-mi"):
        lldbMiExec = lldbtest_config.lldbExec + "-mi"
    if not lldbMiExec:
        if not configuration.shouldSkipBecauseOfCategories(["lldb-mi"]):
            print("The 'lldb-mi' executable cannot be located.  The lldb-mi tests can not be run as a result.")
            configuration.skipCategories.append("lldb-mi")
    else:
        os.environ["LLDBMI_EXEC"] = lldbMiExec

    # Skip printing svn/git information when running in parsable (lit-test compatibility) mode
    if not configuration.svn_silent and not configuration.parsable:
        if os.path.isdir(os.path.join(lldbRootDirectory, '.svn')) and which("svn") is not None:
            pipe = subprocess.Popen([which("svn"), "info", lldbRootDirectory], stdout = subprocess.PIPE)
            configuration.svn_info = pipe.stdout.read()
        elif os.path.isdir(os.path.join(lldbRootDirectory, '.git')) and which("git") is not None:
            pipe = subprocess.Popen([which("git"), "svn", "info", lldbRootDirectory], stdout = subprocess.PIPE)
            configuration.svn_info = pipe.stdout.read()
        if not configuration.noHeaders:
            print(configuration.svn_info)

    lldbPythonDir = None # The directory that contains 'lldb/__init__.py'
    if configuration.lldbFrameworkPath:
        candidatePath = os.path.join(configuration.lldbFrameworkPath, 'Resources', 'Python')
        if os.path.isfile(os.path.join(candidatePath, 'lldb/__init__.py')):
            lldbPythonDir = candidatePath
        if not lldbPythonDir:
            print('Resources/Python/lldb/__init__.py was not found in ' + configuration.lldbFrameworkPath)
            sys.exit(-1)
    else:
        # The '-i' option is used to skip looking for lldb.py in the build tree.
        if configuration.ignore:
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

def visit(prefix, dir, names):
    """Visitor function for os.path.walk(path, visit, arg)."""

    dir_components = set(dir.split(os.sep))
    excluded_components = set(['.svn', '.git'])
    if dir_components.intersection(excluded_components):
        #print("Detected an excluded dir component: %s" % dir)
        return

    for name in names:
        if '.py' == os.path.splitext(name)[1] and name.startswith(prefix):

            if name in configuration.all_tests:
                raise Exception("Found multiple tests with the name %s" % name)
            configuration.all_tests.add(name)

            # Try to match the regexp pattern, if specified.
            if configuration.regexp:
                import re
                if re.search(configuration.regexp, name):
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
            for filterspec in configuration.filters:
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
                    configuration.suite.addTests(
                        unittest2.defaultTestLoader.loadTestsFromName(filterspec, module))
                    continue

            # Forgo this module if the (base, filterspec) combo is invalid
            if configuration.filters and not filtered:
                continue

            # Add either the filtered test case(s) (which is done before) or the entire test class.
            if not filterspec or not filtered:
                # A simple case of just the module name.  Also the failover case
                # from the filterspec branch when the (base, filterspec) combo
                # doesn't make sense.
                configuration.suite.addTests(unittest2.defaultTestLoader.loadTestsFromName(base))


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
    return not (configuration.is_inferior_test_runner or configuration.no_multiprocess_test_runner)

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
    if configuration.lldb_platform_name == 'ios-simulator':
        triple_str = 'x86_64-apple-ios%s' % (getVersionForSDK('iphonesimulator'))
        os.environ['TRIPLE'] = triple_str
        return {'TRIPLE':triple_str}
    return {}

def run_suite():
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
        dosep.main(configuration.output_on_success, configuration.num_threads,
                   configuration.multiprocess_test_subdir,
                   configuration.test_runner_name, configuration.results_formatter_object)
        raise Exception("should never get here")
    elif configuration.is_inferior_test_runner:
        # Shut off Ctrl-C processing in inferiors.  The parallel
        # test runner handles this more holistically.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    setupSysPath()
    configuration.setupCrashInfoHook()

    #
    # If '-l' is specified, do not skip the long running tests.
    if not configuration.skip_long_running_test:
        os.environ["LLDB_SKIP_LONG_RUNNING_TEST"] = "NO"

    # For the time being, let's bracket the test runner within the
    # lldb.SBDebugger.Initialize()/Terminate() pair.
    import lldb

    # Create a singleton SBDebugger in the lldb namespace.
    lldb.DBG = lldb.SBDebugger.Create()

    if configuration.lldb_platform_name:
        print("Setting up remote platform '%s'" % (configuration.lldb_platform_name))
        lldb.remote_platform = lldb.SBPlatform(configuration.lldb_platform_name)
        if not lldb.remote_platform.IsValid():
            print("error: unable to create the LLDB platform named '%s'." % (configuration.lldb_platform_name))
            exitTestSuite(1)
        if configuration.lldb_platform_url:
            # We must connect to a remote platform if a LLDB platform URL was specified
            print("Connecting to remote platform '%s' at '%s'..." % (configuration.lldb_platform_name, configuration.lldb_platform_url))
            platform_connect_options = lldb.SBPlatformConnectOptions(configuration.lldb_platform_url)
            err = lldb.remote_platform.ConnectRemote(platform_connect_options)
            if err.Success():
                print("Connected.")
            else:
                print("error: failed to connect to remote platform using URL '%s': %s" % (configuration.lldb_platform_url, err))
                exitTestSuite(1)
        else:
            configuration.lldb_platform_url = None

    platform_changes = setDefaultTripleForPlatform()
    first = True
    for key in platform_changes:
        if first:
            print("Environment variables setup for platform support:")
            first = False
        print("%s = %s" % (key,platform_changes[key]))

    if configuration.lldb_platform_working_dir:
        print("Setting remote platform working directory to '%s'..." % (configuration.lldb_platform_working_dir))
        lldb.remote_platform.SetWorkingDirectory(configuration.lldb_platform_working_dir)
        lldb.DBG.SetSelectedPlatform(lldb.remote_platform)
    else:
        lldb.remote_platform = None
        configuration.lldb_platform_working_dir = None
        configuration.lldb_platform_url = None

    target_platform = lldb.DBG.GetSelectedPlatform().GetTriple().split('-')[2]

    # By default, both dsym, dwarf and dwo tests are performed.
    # Use @dsym_test, @dwarf_test or @dwo_test decorators, defined in lldbtest.py, to mark a test as
    # a dsym, dwarf or dwo test.  Use '-N dsym', '-N dwarf' or '-N dwo' to exclude dsym, dwarf or
    # dwo tests from running.
    configuration.dont_do_dsym_test = configuration.dont_do_dsym_test \
        or any(platform in target_platform for platform in ["linux", "freebsd", "windows"])
    configuration.dont_do_dwo_test = configuration.dont_do_dwo_test \
        or any(platform in target_platform for platform in ["darwin", "macosx", "ios"])

    # Don't do debugserver tests on everything except OS X.
    configuration.dont_do_debugserver_test = "linux" in target_platform or "freebsd" in target_platform or "windows" in target_platform

    # Don't do lldb-server (llgs) tests on anything except Linux.
    configuration.dont_do_llgs_test = not ("linux" in target_platform)

    #
    # Walk through the testdirs while collecting tests.
    #
    for testdir in configuration.testdirs:
        for (dirpath, dirnames, filenames) in os.walk(testdir):
            visit('Test', dirpath, filenames)

    #
    # Now that we have loaded all the test cases, run the whole test suite.
    #

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
    if not configuration.sdir_name:
        configuration.sdir_name = timestamp_started
    os.environ["LLDB_SESSION_DIRNAME"] = os.path.join(os.getcwd(), configuration.sdir_name)

    if not configuration.noHeaders:
        sys.stderr.write("\nSession logs for test failures/errors/unexpected successes"
                         " will go into directory '%s'\n" % configuration.sdir_name)
        sys.stderr.write("Command invoked: %s\n" % getMyCommandLine())

    if not os.path.isdir(configuration.sdir_name):
        try:
            os.mkdir(configuration.sdir_name)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    where_to_save_session = os.getcwd()
    fname = os.path.join(configuration.sdir_name, "TestStarted-%d" % os.getpid())
    with open(fname, "w") as f:
        print("Test started at: %s\n" % timestamp_started, file=f)
        print(configuration.svn_info, file=f)
        print("Command invoked: %s\n" % getMyCommandLine(), file=f)

    #
    # Invoke the default TextTestRunner to run the test suite, possibly iterating
    # over different configurations.
    #

    iterArchs = False
    iterCompilers = False

    if isinstance(configuration.archs, list) and len(configuration.archs) >= 1:
        iterArchs = True

    #
    # Add some intervention here to sanity check that the compilers requested are sane.
    # If found not to be an executable program, the invalid one is dropped from the list.
    for i in range(len(configuration.compilers)):
        c = configuration.compilers[i]
        if which(c):
            continue
        else:
            if sys.platform.startswith("darwin"):
                pipe = subprocess.Popen(['xcrun', '-find', c], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
                cmd_output = pipe.stdout.read()
                if cmd_output:
                    if "not found" in cmd_output:
                        print("dropping %s from the compilers used" % c)
                        configuration.compilers.remove(i)
                    else:
                        configuration.compilers[i] = cmd_output.split('\n')[0]
                        print("'xcrun -find %s' returning %s" % (c, configuration.compilers[i]))

    if not configuration.parsable:
        print("compilers=%s" % str(configuration.compilers))

    if not configuration.compilers or len(configuration.compilers) == 0:
        print("No eligible compiler found, exiting.")
        exitTestSuite(1)

    if isinstance(configuration.compilers, list) and len(configuration.compilers) >= 1:
        iterCompilers = True

    # Make a shallow copy of sys.path, we need to manipulate the search paths later.
    # This is only necessary if we are relocated and with different configurations.
    if configuration.rdir:
        old_sys_path = sys.path[:]
    # If we iterate on archs or compilers, there is a chance we want to split stderr/stdout.
    if iterArchs or iterCompilers:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        new_stderr = None
        new_stdout = None

    # Iterating over all possible architecture and compiler combinations.
    for ia in range(len(configuration.archs) if iterArchs else 1):
        archConfig = ""
        if iterArchs:
            os.environ["ARCH"] = configuration.archs[ia]
            archConfig = "arch=%s" % configuration.archs[ia]
        for ic in range(len(configuration.compilers) if iterCompilers else 1):
            if iterCompilers:
                os.environ["CC"] = configuration.compilers[ic]
                configString = "%s compiler=%s" % (archConfig, configuration.compilers[ic])
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

                # If we specified a relocated directory to run the test suite, do
                # the extra housekeeping to copy the testdirs to a configStringified
                # directory and to update sys.path before invoking the test runner.
                # The purpose is to separate the configuration-specific directories
                # from each other.
                if configuration.rdir:
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
                    if len(configuration.testdirs) == 1 and os.path.basename(configuration.testdirs[0]) == 'test':
                        os.environ["LLDB_TEST"] = os.path.join(newrdir, 'test')
                    else:
                        os.environ["LLDB_TEST"] = newrdir

                    # And update the Python search paths for modules.
                    sys.path = [x.replace(rdir, newrdir, 1) for x in old_sys_path]

                # Output the configuration.
                if not configuration.parsable:
                    sys.stderr.write("\nConfiguration: " + configString + "\n")

            #print("sys.stderr name is", sys.stderr.name)
            #print("sys.stdout name is", sys.stdout.name)

            # First, write out the number of collected test cases.
            if not configuration.parsable:
                sys.stderr.write(configuration.separator + "\n")
                sys.stderr.write("Collected %d test%s\n\n"
                                 % (configuration.suite.countTestCases(),
                                    configuration.suite.countTestCases() != 1 and "s" or ""))

            if configuration.parsable:
                v = 0
            elif configuration.progress_bar:
                v = 1
            else:
                v = configuration.verbose

            # Invoke the test runner.
            if configuration.count == 1:
                result = unittest2.TextTestRunner(stream=sys.stderr,
                                                  verbosity=v,
                                                  failfast=configuration.failfast,
                                                  resultclass=test_result.LLDBTestResult).run(configuration.suite)
            else:
                # We are invoking the same test suite more than once.  In this case,
                # mark __ignore_singleton__ flag as True so the signleton pattern is
                # not enforced.
                test_result.LLDBTestResult.__ignore_singleton__ = True
                for i in range(count):
               
                    result = unittest2.TextTestRunner(stream=sys.stderr,
                                                      verbosity=v,
                                                      failfast=configuration.failfast,
                                                      resultclass=test_result.LLDBTestResult).run(configuration.suite)

            configuration.failed = configuration.failed or not result.wasSuccessful()

    if configuration.sdir_has_content and not configuration.parsable:
        sys.stderr.write("Session logs for test failures/errors/unexpected successes"
                         " can be found in directory '%s'\n" % configuration.sdir_name)

    if configuration.useCategories and len(configuration.failuresPerCategory) > 0:
        sys.stderr.write("Failures per category:\n")
        for category in configuration.failuresPerCategory:
            sys.stderr.write("%s - %d\n" % (category, configuration.failuresPerCategory[category]))

    os.chdir(where_to_save_session)
    fname = os.path.join(configuration.sdir_name, "TestFinished-%d" % os.getpid())
    with open(fname, "w") as f:
        print("Test finished at: %s\n" % datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"), file=f)

    # Terminate the test suite if ${LLDB_TESTSUITE_FORCE_FINISH} is defined.
    # This should not be necessary now.
    if ("LLDB_TESTSUITE_FORCE_FINISH" in os.environ):
        print("Terminating Test suite...")
        subprocess.Popen(["/bin/sh", "-c", "kill %s; exit 0" % (os.getpid())])

    # Exiting.
    exitTestSuite(configuration.failed)

if __name__ == "__main__":
    print(__file__ + " is for use as a module only.  It should not be run as a standalone script.")
    sys.exit(-1)
