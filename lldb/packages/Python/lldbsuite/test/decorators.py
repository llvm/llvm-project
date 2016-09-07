from __future__ import absolute_import
from __future__ import print_function

# System modules
from distutils.version import LooseVersion, StrictVersion
from functools import wraps
import os
import re
import sys
import tempfile

# Third-party modules
import six
import unittest2

# LLDB modules
import use_lldb_suite

import lldb
from . import configuration
from . import test_categories
from lldbsuite.test_event.event_builder import EventBuilder
from lldbsuite.support import funcutils
from lldbsuite.test import lldbplatform
from lldbsuite.test import lldbplatformutil


class DecorateMode:
    Skip, Xfail = range(2)


# You can use no_match to reverse the test of the conditional that is used to match keyword
# arguments in the skip / xfail decorators.  If oslist=["windows", "linux"] skips windows
# and linux, oslist=no_match(["windows", "linux"]) skips *unless* windows
# or linux.
class no_match:

    def __init__(self, item):
        self.item = item


def _check_expected_version(comparison, expected, actual):
    def fn_leq(x, y): return x <= y

    def fn_less(x, y): return x < y

    def fn_geq(x, y): return x >= y

    def fn_greater(x, y): return x > y

    def fn_eq(x, y): return x == y

    def fn_neq(x, y): return x != y

    op_lookup = {
        "==": fn_eq,
        "=": fn_eq,
        "!=": fn_neq,
        "<>": fn_neq,
        ">": fn_greater,
        "<": fn_less,
        ">=": fn_geq,
        "<=": fn_leq
    }
    expected_str = '.'.join([str(x) for x in expected])
    actual_str = '.'.join([str(x) for x in actual])

    return op_lookup[comparison](
        LooseVersion(actual_str),
        LooseVersion(expected_str))


def _match_decorator_property(expected, actual):
    if actual is None or expected is None:
        return True

    if isinstance(expected, no_match):
        return not _match_decorator_property(expected.item, actual)
    elif isinstance(expected, (re._pattern_type,) + six.string_types):
        return re.search(expected, actual) is not None
    elif hasattr(expected, "__iter__"):
        return any([x is not None and _match_decorator_property(x, actual)
                    for x in expected])
    else:
        return expected == actual


def expectedFailure(expected_fn, bugnumber=None):
    def expectedFailure_impl(func):
        if isinstance(func, type) and issubclass(func, unittest2.TestCase):
            raise Exception(
                "Decorator can only be used to decorate a test method")

        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            if funcutils.requires_self(expected_fn):
                xfail_reason = expected_fn(self)
            else:
                xfail_reason = expected_fn()
            if xfail_reason is not None:
                if configuration.results_formatter_object is not None:
                    # Mark this test as expected to fail.
                    configuration.results_formatter_object.handle_event(
                        EventBuilder.event_for_mark_test_expected_failure(self))
                xfail_func = unittest2.expectedFailure(func)
                xfail_func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        return wrapper
    # Some decorators can be called both with no arguments (e.g. @expectedFailureWindows)
    # or with arguments (e.g. @expectedFailureWindows(compilers=['gcc'])).  When called
    # the first way, the first argument will be the actual function because decorators are
    # weird like that.  So this is basically a check that says "which syntax was the original
    # function decorated with?"
    if six.callable(bugnumber):
        return expectedFailure_impl(bugnumber)
    else:
        return expectedFailure_impl


def skipTestIfFn(expected_fn, bugnumber=None):
    def skipTestIfFn_impl(func):
        if isinstance(func, type) and issubclass(func, unittest2.TestCase):
            raise Exception(
                "@skipTestIfFn can only be used to decorate a test method")

        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            if funcutils.requires_self(expected_fn):
                reason = expected_fn(self)
            else:
                reason = expected_fn()

            if reason is not None:
                self.skipTest(reason)
            else:
                func(*args, **kwargs)
        return wrapper

    # Some decorators can be called both with no arguments (e.g. @expectedFailureWindows)
    # or with arguments (e.g. @expectedFailureWindows(compilers=['gcc'])).  When called
    # the first way, the first argument will be the actual function because decorators are
    # weird like that.  So this is basically a check that says "how was the
    # decorator used"
    if six.callable(bugnumber):
        return skipTestIfFn_impl(bugnumber)
    else:
        return skipTestIfFn_impl


def _decorateTest(mode,
                  bugnumber=None, oslist=None, hostoslist=None,
                  compiler=None, compiler_version=None,
                  archs=None, triple=None,
                  debug_info=None,
                  swig_version=None, py_version=None,
                  remote=None):
    def fn(self):
        skip_for_os = _match_decorator_property(
            lldbplatform.translate(oslist), self.getPlatform())
        skip_for_hostos = _match_decorator_property(
            lldbplatform.translate(hostoslist),
            lldbplatformutil.getHostPlatform())
        skip_for_compiler = _match_decorator_property(
            compiler, self.getCompiler()) and self.expectedCompilerVersion(compiler_version)
        skip_for_arch = _match_decorator_property(
            archs, self.getArchitecture())
        skip_for_debug_info = _match_decorator_property(
            debug_info, self.debug_info)
        skip_for_triple = _match_decorator_property(
            triple, lldb.DBG.GetSelectedPlatform().GetTriple())
        skip_for_remote = _match_decorator_property(
            remote, lldb.remote_platform is not None)

        skip_for_swig_version = (
            swig_version is None) or (
            not hasattr(
                lldb,
                'swig_version')) or (
                _check_expected_version(
                    swig_version[0],
                    swig_version[1],
                    lldb.swig_version))
        skip_for_py_version = (
            py_version is None) or _check_expected_version(
            py_version[0], py_version[1], sys.version_info)

        # For the test to be skipped, all specified (e.g. not None) parameters must be True.
        # An unspecified parameter means "any", so those are marked skip by default.  And we skip
        # the final test if all conditions are True.
        conditions = [(oslist, skip_for_os, "target o/s"),
                      (hostoslist, skip_for_hostos, "host o/s"),
                      (compiler, skip_for_compiler, "compiler or version"),
                      (archs, skip_for_arch, "architecture"),
                      (debug_info, skip_for_debug_info, "debug info format"),
                      (triple, skip_for_triple, "target triple"),
                      (swig_version, skip_for_swig_version, "swig version"),
                      (py_version, skip_for_py_version, "python version"),
                      (remote, skip_for_remote, "platform locality (remote/local)")]
        reasons = []
        final_skip_result = True
        for this_condition in conditions:
            final_skip_result = final_skip_result and this_condition[1]
            if this_condition[0] is not None and this_condition[1]:
                reasons.append(this_condition[2])
        reason_str = None
        if final_skip_result:
            mode_str = {
                DecorateMode.Skip: "skipping",
                DecorateMode.Xfail: "xfailing"}[mode]
            if len(reasons) > 0:
                reason_str = ",".join(reasons)
                reason_str = "{} due to the following parameter(s): {}".format(
                    mode_str, reason_str)
            else:
                reason_str = "{} unconditionally"
            if bugnumber is not None and not six.callable(bugnumber):
                reason_str = reason_str + " [" + str(bugnumber) + "]"
        return reason_str

    if mode == DecorateMode.Skip:
        return skipTestIfFn(fn, bugnumber)
    elif mode == DecorateMode.Xfail:
        return expectedFailure(fn, bugnumber)
    else:
        return None

# provide a function to xfail on defined oslist, compiler version, and archs
# if none is specified for any argument, that argument won't be checked and thus means for all
# for example,
# @expectedFailureAll, xfail for all platform/compiler/arch,
# @expectedFailureAll(compiler='gcc'), xfail for gcc on all platform/architecture
# @expectedFailureAll(bugnumber, ["linux"], "gcc", ['>=', '4.9'], ['i386']), xfail for gcc>=4.9 on linux with i386


def expectedFailureAll(bugnumber=None,
                       oslist=None, hostoslist=None,
                       compiler=None, compiler_version=None,
                       archs=None, triple=None,
                       debug_info=None,
                       swig_version=None, py_version=None,
                       remote=None):
    return _decorateTest(DecorateMode.Xfail,
                         bugnumber=bugnumber,
                         oslist=oslist, hostoslist=hostoslist,
                         compiler=compiler, compiler_version=compiler_version,
                         archs=archs, triple=triple,
                         debug_info=debug_info,
                         swig_version=swig_version, py_version=py_version,
                         remote=remote)


# provide a function to skip on defined oslist, compiler version, and archs
# if none is specified for any argument, that argument won't be checked and thus means for all
# for example,
# @skipIf, skip for all platform/compiler/arch,
# @skipIf(compiler='gcc'), skip for gcc on all platform/architecture
# @skipIf(bugnumber, ["linux"], "gcc", ['>=', '4.9'], ['i386']), skip for gcc>=4.9 on linux with i386
def skipIf(bugnumber=None,
           oslist=None, hostoslist=None,
           compiler=None, compiler_version=None,
           archs=None, triple=None,
           debug_info=None,
           swig_version=None, py_version=None,
           remote=None):
    return _decorateTest(DecorateMode.Skip,
                         bugnumber=bugnumber,
                         oslist=oslist, hostoslist=hostoslist,
                         compiler=compiler, compiler_version=compiler_version,
                         archs=archs, triple=triple,
                         debug_info=debug_info,
                         swig_version=swig_version, py_version=py_version,
                         remote=remote)


def _skip_for_android(reason, api_levels, archs):
    def impl(obj):
        result = lldbplatformutil.match_android_device(
            obj.getArchitecture(), valid_archs=archs, valid_api_levels=api_levels)
        return reason if result else None
    return impl


def add_test_categories(cat):
    """Add test categories to a TestCase method"""
    cat = test_categories.validate(cat, True)

    def impl(func):
        if isinstance(func, type) and issubclass(func, unittest2.TestCase):
            raise Exception(
                "@add_test_categories can only be used to decorate a test method")
        if hasattr(func, "categories"):
            cat.extend(func.categories)
        func.categories = cat
        return func

    return impl


def benchmarks_test(func):
    """Decorate the item as a benchmarks test."""
    def should_skip_benchmarks_test():
        return "benchmarks test"

    # Mark this function as such to separate them from the regular tests.
    result = skipTestIfFn(should_skip_benchmarks_test)(func)
    result.__benchmarks_test__ = True
    return result


def no_debug_info_test(func):
    """Decorate the item as a test what don't use any debug info. If this annotation is specified
       then the test runner won't generate a separate test for each debug info format. """
    if isinstance(func, type) and issubclass(func, unittest2.TestCase):
        raise Exception(
            "@no_debug_info_test can only be used to decorate a test method")

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    # Mark this function as such to separate them from the regular tests.
    wrapper.__no_debug_info_test__ = True
    return wrapper


def debugserver_test(func):
    """Decorate the item as a debugserver test."""
    def should_skip_debugserver_test():
        return "debugserver tests" if configuration.dont_do_debugserver_test else None
    return skipTestIfFn(should_skip_debugserver_test)(func)


def llgs_test(func):
    """Decorate the item as a lldb-server test."""
    def should_skip_llgs_tests():
        return "llgs tests" if configuration.dont_do_llgs_test else None
    return skipTestIfFn(should_skip_llgs_tests)(func)


def not_remote_testsuite_ready(func):
    """Decorate the item as a test which is not ready yet for remote testsuite."""
    def is_remote():
        return "Not ready for remote testsuite" if lldb.remote_platform else None
    return skipTestIfFn(is_remote)(func)


def expectedFailureOS(
        oslist,
        bugnumber=None,
        compilers=None,
        debug_info=None,
        archs=None):
    return expectedFailureAll(
        oslist=oslist,
        bugnumber=bugnumber,
        compiler=compilers,
        archs=archs,
        debug_info=debug_info)


def expectedFailureDarwin(bugnumber=None, compilers=None, debug_info=None):
    # For legacy reasons, we support both "darwin" and "macosx" as OS X
    # triples.
    return expectedFailureOS(
        lldbplatform.darwin_all,
        bugnumber,
        compilers,
        debug_info=debug_info)


def expectedFailureAndroid(bugnumber=None, api_levels=None, archs=None):
    """ Mark a test as xfail for Android.

    Arguments:
        bugnumber - The LLVM pr associated with the problem.
        api_levels - A sequence of numbers specifying the Android API levels
            for which a test is expected to fail. None means all API level.
        arch - A sequence of architecture names specifying the architectures
            for which a test is expected to fail. None means all architectures.
    """
    return expectedFailure(
        _skip_for_android(
            "xfailing on android",
            api_levels,
            archs),
        bugnumber)

# Flakey tests get two chances to run. If they fail the first time round, the result formatter
# makes sure it is run one more time.


def expectedFlakey(expected_fn, bugnumber=None):
    def expectedFailure_impl(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            if expected_fn(self):
                # Send event marking test as explicitly eligible for rerunning.
                if configuration.results_formatter_object is not None:
                    # Mark this test as rerunnable.
                    configuration.results_formatter_object.handle_event(
                        EventBuilder.event_for_mark_test_rerun_eligible(self))
            func(*args, **kwargs)
        return wrapper
    # Some decorators can be called both with no arguments (e.g. @expectedFailureWindows)
    # or with arguments (e.g. @expectedFailureWindows(compilers=['gcc'])).  When called
    # the first way, the first argument will be the actual function because decorators are
    # weird like that.  So this is basically a check that says "which syntax was the original
    # function decorated with?"
    if six.callable(bugnumber):
        return expectedFailure_impl(bugnumber)
    else:
        return expectedFailure_impl


def expectedFlakeyDwarf(bugnumber=None):
    def fn(self):
        return self.debug_info == "dwarf"
    return expectedFlakey(fn, bugnumber)


def expectedFlakeyDsym(bugnumber=None):
    def fn(self):
        return self.debug_info == "dwarf"
    return expectedFlakey(fn, bugnumber)


def expectedFlakeyOS(oslist, bugnumber=None, compilers=None):
    def fn(self):
        return (self.getPlatform() in oslist and
                self.expectedCompiler(compilers))
    return expectedFlakey(fn, bugnumber)


def expectedFlakeyDarwin(bugnumber=None, compilers=None):
    # For legacy reasons, we support both "darwin" and "macosx" as OS X
    # triples.
    return expectedFlakeyOS(
        lldbplatformutil.getDarwinOSTriples(),
        bugnumber,
        compilers)


def expectedFlakeyFreeBSD(bugnumber=None, compilers=None):
    return expectedFlakeyOS(['freebsd'], bugnumber, compilers)


def expectedFlakeyLinux(bugnumber=None, compilers=None):
    return expectedFlakeyOS(['linux'], bugnumber, compilers)


def expectedFlakeyNetBSD(bugnumber=None, compilers=None):
    return expectedFlakeyOS(['netbsd'], bugnumber, compilers)


def expectedFlakeyCompiler(compiler, compiler_version=None, bugnumber=None):
    if compiler_version is None:
        compiler_version = ['=', None]

    def fn(self):
        return compiler in self.getCompiler() and self.expectedCompilerVersion(compiler_version)
    return expectedFlakey(fn, bugnumber)

# @expectedFlakeyClang('bugnumber', ['<=', '3.4'])


def expectedFlakeyClang(bugnumber=None, compiler_version=None):
    return expectedFlakeyCompiler('clang', compiler_version, bugnumber)

# @expectedFlakeyGcc('bugnumber', ['<=', '3.4'])


def expectedFlakeyGcc(bugnumber=None, compiler_version=None):
    return expectedFlakeyCompiler('gcc', compiler_version, bugnumber)


def expectedFlakeyAndroid(bugnumber=None, api_levels=None, archs=None):
    return expectedFlakey(
        _skip_for_android(
            "flakey on android",
            api_levels,
            archs),
        bugnumber)


def skipIfRemote(func):
    """Decorate the item to skip tests if testing remotely."""
    def is_remote():
        return "skip on remote platform" if lldb.remote_platform else None
    return skipTestIfFn(is_remote)(func)


def skipIfRemoteDueToDeadlock(func):
    """Decorate the item to skip tests if testing remotely due to the test deadlocking."""
    def is_remote():
        return "skip on remote platform (deadlocks)" if lldb.remote_platform else None
    return skipTestIfFn(is_remote)(func)


def skipIfNoSBHeaders(func):
    """Decorate the item to mark tests that should be skipped when LLDB is built with no SB API headers."""
    def are_sb_headers_missing():
        if lldbplatformutil.getHostPlatform() == 'darwin':
            header = os.path.join(
                os.environ["LLDB_LIB_DIR"],
                'LLDB.framework',
                'Versions',
                'Current',
                'Headers',
                'LLDB.h')
        else:
            header = os.path.join(
                os.environ["LLDB_SRC"],
                "include",
                "lldb",
                "API",
                "LLDB.h")
        if not os.path.exists(header):
            return "skip because LLDB.h header not found"
        return None

    return skipTestIfFn(are_sb_headers_missing)(func)


def skipIfiOSSimulator(func):
    """Decorate the item to skip tests that should be skipped on the iOS Simulator."""
    def is_ios_simulator():
        return "skip on the iOS Simulator" if configuration.lldb_platform_name == 'ios-simulator' else None
    return skipTestIfFn(is_ios_simulator)(func)


def skipIfFreeBSD(func):
    """Decorate the item to skip tests that should be skipped on FreeBSD."""
    return skipIfPlatform(["freebsd"])(func)


def skipIfNetBSD(func):
    """Decorate the item to skip tests that should be skipped on NetBSD."""
    return skipIfPlatform(["netbsd"])(func)


def skipIfDarwin(func):
    """Decorate the item to skip tests that should be skipped on Darwin."""
    return skipIfPlatform(
        lldbplatform.translate(
            lldbplatform.darwin_all))(func)


def skipIfLinux(func):
    """Decorate the item to skip tests that should be skipped on Linux."""
    return skipIfPlatform(["linux"])(func)


def skipIfWindows(func):
    """Decorate the item to skip tests that should be skipped on Windows."""
    return skipIfPlatform(["windows"])(func)


def skipUnlessWindows(func):
    """Decorate the item to skip tests that should be skipped on any non-Windows platform."""
    return skipUnlessPlatform(["windows"])(func)


def skipUnlessDarwin(func):
    """Decorate the item to skip tests that should be skipped on any non Darwin platform."""
    return skipUnlessPlatform(lldbplatformutil.getDarwinOSTriples())(func)


def skipUnlessGoInstalled(func):
    """Decorate the item to skip tests when no Go compiler is available."""

    def is_go_missing(self):
        compiler = self.getGoCompilerVersion()
        if not compiler:
            return "skipping because go compiler not found"
        match_version = re.search(r"(\d+\.\d+(\.\d+)?)", compiler)
        if not match_version:
            # Couldn't determine version.
            return "skipping because go version could not be parsed out of {}".format(
                compiler)
        else:
            min_strict_version = StrictVersion("1.4.0")
            compiler_strict_version = StrictVersion(match_version.group(1))
            if compiler_strict_version < min_strict_version:
                return "skipping because available version ({}) does not meet minimum required version ({})".format(
                    compiler_strict_version, min_strict_version)
        return None
    return skipTestIfFn(is_go_missing)(func)


def skipIfHostIncompatibleWithRemote(func):
    """Decorate the item to skip tests if binaries built on this host are incompatible."""

    def is_host_incompatible_with_remote(self):
        host_arch = self.getLldbArchitecture()
        host_platform = lldbplatformutil.getHostPlatform()
        target_arch = self.getArchitecture()
        target_platform = 'darwin' if self.platformIsDarwin() else self.getPlatform()
        if not (target_arch == 'x86_64' and host_arch ==
                'i386') and host_arch != target_arch:
            return "skipping because target %s is not compatible with host architecture %s" % (
                target_arch, host_arch)
        elif target_platform != host_platform:
            return "skipping because target is %s but host is %s" % (
                target_platform, host_platform)
        return None
    return skipTestIfFn(is_host_incompatible_with_remote)(func)


def skipIfPlatform(oslist):
    """Decorate the item to skip tests if running on one of the listed platforms."""
    # This decorator cannot be ported to `skipIf` yet because it is used on entire
    # classes, which `skipIf` explicitly forbids.
    return unittest2.skipIf(lldbplatformutil.getPlatform() in oslist,
                            "skip on %s" % (", ".join(oslist)))


def skipUnlessPlatform(oslist):
    """Decorate the item to skip tests unless running on one of the listed platforms."""
    # This decorator cannot be ported to `skipIf` yet because it is used on entire
    # classes, which `skipIf` explicitly forbids.
    return unittest2.skipUnless(lldbplatformutil.getPlatform() in oslist,
                                "requires one of %s" % (", ".join(oslist)))


def skipIfTargetAndroid(api_levels=None, archs=None):
    """Decorator to skip tests when the target is Android.

    Arguments:
        api_levels - The API levels for which the test should be skipped. If
            it is None, then the test will be skipped for all API levels.
        arch - A sequence of architecture names specifying the architectures
            for which a test is skipped. None means all architectures.
    """
    return skipTestIfFn(
        _skip_for_android(
            "skipping for android",
            api_levels,
            archs))


def skipUnlessCompilerRt(func):
    """Decorate the item to skip tests if testing remotely."""
    def is_compiler_rt_missing():
        compilerRtPath = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "llvm",
            "projects",
            "compiler-rt")
        return "compiler-rt not found" if not os.path.exists(
            compilerRtPath) else None
    return skipTestIfFn(is_compiler_rt_missing)(func)


def skipUnlessThreadSanitizer(func):
    """Decorate the item to skip test unless Clang -fsanitize=thread is supported."""

    def is_compiler_clang_with_thread_sanitizer(self):
        compiler_path = self.getCompiler()
        compiler = os.path.basename(compiler_path)
        if not compiler.startswith("clang"):
            return "Test requires clang as compiler"
        f = tempfile.NamedTemporaryFile()
        cmd = "echo 'int main() {}' | %s -x c -o %s -" % (compiler_path, f.name)
        if os.popen(cmd).close() is not None:
            return None  # The compiler cannot compile at all, let's *not* skip the test
        cmd = "echo 'int main() {}' | %s -fsanitize=thread -x c -o %s -" % (compiler_path, f.name)
        if os.popen(cmd).close() is not None:
            return "Compiler cannot compile with -fsanitize=thread"
        return None
    return skipTestIfFn(is_compiler_clang_with_thread_sanitizer)(func)
