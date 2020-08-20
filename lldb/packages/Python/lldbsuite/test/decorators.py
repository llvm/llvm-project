from __future__ import absolute_import

# System modules
from distutils.version import LooseVersion
from functools import wraps
import os
import platform
import re
import sys
import tempfile
import subprocess

# Third-party modules
import six
import unittest2

# LLDB modules
import lldb
from . import configuration
from . import test_categories
from . import lldbtest_config
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


_re_pattern_type = type(re.compile(''))
def _match_decorator_property(expected, actual):
    if actual is None or expected is None:
        return True

    if isinstance(expected, no_match):
        return not _match_decorator_property(expected.item, actual)
    elif isinstance(expected, (_re_pattern_type,) + six.string_types):
        return re.search(expected, actual) is not None
    elif hasattr(expected, "__iter__"):
        return any([x is not None and _match_decorator_property(x, actual)
                    for x in expected])
    else:
        return expected == actual


def expectedFailure(func, bugnumber=None):
    return unittest2.expectedFailure(func)

def expectedFailureIfFn(expected_fn, bugnumber=None):
    def expectedFailure_impl(func):
        if isinstance(func, type) and issubclass(func, unittest2.TestCase):
            raise Exception(
                "Decorator can only be used to decorate a test method")

        @wraps(func)
        def wrapper(*args, **kwargs):
            xfail_reason = expected_fn(*args, **kwargs)
            if xfail_reason is not None:
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
                return func(*args, **kwargs)
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
                  macos_version=None,
                  remote=None, dwarf_version=None,
                  setting=None):
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
            debug_info, self.getDebugInfo())
        skip_for_triple = _match_decorator_property(
            triple, lldb.selected_platform.GetTriple())
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
        skip_for_macos_version = (macos_version is None) or (
            (platform.mac_ver()[0] != "") and (_check_expected_version(
                macos_version[0],
                macos_version[1],
                platform.mac_ver()[0])))
        skip_for_dwarf_version = (dwarf_version is None) or (
            _check_expected_version(dwarf_version[0], dwarf_version[1],
                                    self.getDwarfVersion()))
        skip_for_setting = (setting is None) or (
            setting in configuration.settings)

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
                      (macos_version, skip_for_macos_version, "macOS version"),
                      (remote, skip_for_remote, "platform locality (remote/local)"),
                      (dwarf_version, skip_for_dwarf_version, "dwarf version"),
                      (setting, skip_for_setting, "setting")]
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
        return expectedFailureIfFn(fn, bugnumber)
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
                       macos_version=None,
                       remote=None, dwarf_version=None,
                       setting=None):
    return _decorateTest(DecorateMode.Xfail,
                         bugnumber=bugnumber,
                         oslist=oslist, hostoslist=hostoslist,
                         compiler=compiler, compiler_version=compiler_version,
                         archs=archs, triple=triple,
                         debug_info=debug_info,
                         swig_version=swig_version, py_version=py_version,
                         macos_version=None,
                         remote=remote,dwarf_version=dwarf_version,
                         setting=setting)


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
           macos_version=None,
           remote=None, dwarf_version=None,
           setting=None):
    return _decorateTest(DecorateMode.Skip,
                         bugnumber=bugnumber,
                         oslist=oslist, hostoslist=hostoslist,
                         compiler=compiler, compiler_version=compiler_version,
                         archs=archs, triple=triple,
                         debug_info=debug_info,
                         swig_version=swig_version, py_version=py_version,
                         macos_version=macos_version,
                         remote=remote, dwarf_version=dwarf_version,
                         setting=setting)


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
        try:
            if hasattr(func, "categories"):
                cat.extend(func.categories)
            setattr(func, "categories", cat)
        except AttributeError:
            raise Exception('Cannot assign categories to inline tests.')

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

def apple_simulator_test(platform):
    """
    Decorate the test as a test requiring a simulator for a specific platform.

    Consider that a simulator is available if you have the corresponding SDK installed.
    The SDK identifiers for simulators are iphonesimulator, appletvsimulator, watchsimulator
    """
    def should_skip_simulator_test():
        if lldbplatformutil.getHostPlatform() != 'darwin':
            return "simulator tests are run only on darwin hosts"
        try:
            DEVNULL = open(os.devnull, 'w')
            output = subprocess.check_output(["xcodebuild", "-showsdks"], stderr=DEVNULL).decode("utf-8")
            if re.search('%ssimulator' % platform, output):
                return None
            else:
                return "%s simulator is not supported on this system." % platform
        except subprocess.CalledProcessError:
            return "Simulators are unsupported on this system (xcodebuild failed)"

    return skipTestIfFn(should_skip_simulator_test)


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


def expectedFailureDarwin(bugnumber=None, compilers=None, debug_info=None, archs=None):
    # For legacy reasons, we support both "darwin" and "macosx" as OS X
    # triples.
    return expectedFailureOS(
        lldbplatform.darwin_all,
        bugnumber,
        compilers,
        debug_info=debug_info,
        archs=archs)


def expectedFailureAndroid(bugnumber=None, api_levels=None, archs=None):
    """ Mark a test as xfail for Android.

    Arguments:
        bugnumber - The LLVM pr associated with the problem.
        api_levels - A sequence of numbers specifying the Android API levels
            for which a test is expected to fail. None means all API level.
        arch - A sequence of architecture names specifying the architectures
            for which a test is expected to fail. None means all architectures.
    """
    return expectedFailureIfFn(
        _skip_for_android(
            "xfailing on android",
            api_levels,
            archs),
        bugnumber)


def expectedFailureNetBSD(bugnumber=None):
    return expectedFailureOS(
        ['netbsd'],
        bugnumber)

# TODO: This decorator does not do anything. Remove it.
def expectedFlakey(expected_fn, bugnumber=None):
    def expectedFailure_impl(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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


def expectedFlakeyAndroid(bugnumber=None, api_levels=None, archs=None):
    return expectedFlakey(
        _skip_for_android(
            "flakey on android",
            api_levels,
            archs),
        bugnumber)

def skipIfOutOfTreeDebugserver(func):
    """Decorate the item to skip tests if using an out-of-tree debugserver."""
    def is_out_of_tree_debugserver():
        return "out-of-tree debugserver" if lldbtest_config.out_of_tree_debugserver else None
    return skipTestIfFn(is_out_of_tree_debugserver)(func)

def skipIfRemote(func):
    """Decorate the item to skip tests if testing remotely."""
    return unittest2.skipIf(lldb.remote_platform, "skip on remote platform")(func)


def skipIfNoSBHeaders(func):
    """Decorate the item to mark tests that should be skipped when LLDB is built with no SB API headers."""
    def are_sb_headers_missing():
        if lldb.remote_platform:
            return "skip because SBHeaders tests make no sense remotely"

        if lldbplatformutil.getHostPlatform() == 'darwin':
            header = os.path.join(
                os.environ["LLDB_LIB_DIR"],
                'LLDB.framework',
                'Versions',
                'Current',
                'Headers',
                'LLDB.h')
            if os.path.exists(header):
                return None

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


def skipIfRosetta(bugnumber):
    """Skip a test when running the testsuite on macOS under the Rosetta translation layer."""
    def is_running_rosetta(self):
        if lldbplatformutil.getPlatform() in ['darwin', 'macosx']:
            if (platform.uname()[5] == "arm") and (self.getArchitecture() == "x86_64"):
                return "skipped under Rosetta"
        return None
    return skipTestIfFn(is_running_rosetta)

def skipIfiOSSimulator(func):
    """Decorate the item to skip tests that should be skipped on the iOS Simulator."""
    def is_ios_simulator():
        return "skip on the iOS Simulator" if configuration.lldb_platform_name == 'ios-simulator' else None
    return skipTestIfFn(is_ios_simulator)(func)

def skipIfiOS(func):
    return skipIfPlatform(["ios"])(func)

def skipIftvOS(func):
    return skipIfPlatform(["tvos"])(func)

def skipIfwatchOS(func):
    return skipIfPlatform(["watchos"])(func)

def skipIfbridgeOS(func):
    return skipIfPlatform(["bridgeos"])(func)

def skipIfDarwinEmbedded(func):
    """Decorate the item to skip tests that should be skipped on Darwin armv7/arm64 targets."""
    return skipIfPlatform(
        lldbplatform.translate(
            lldbplatform.darwin_embedded))(func)

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

def skipUnlessTargetAndroid(func):
    return unittest2.skipUnless(lldbplatformutil.target_is_android(),
                                "requires target to be Android")(func)


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
        if target_platform != host_platform:
            return "skipping because target is %s but host is %s" % (
                target_platform, host_platform)
        if lldbplatformutil.match_android_device(target_arch):
            return "skipping because target is android"
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

def skipUnlessArch(arch):
    """Decorate the item to skip tests unless running on the specified architecture."""

    def arch_doesnt_match(self):
        target_arch = self.getArchitecture()
        if arch != target_arch:
            return "Test only runs on " + arch + ", but target arch is " + target_arch
        return None

    return skipTestIfFn(arch_doesnt_match)

def skipIfTargetAndroid(bugnumber=None, api_levels=None, archs=None):
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
            archs),
        bugnumber)

def skipUnlessSupportedTypeAttribute(attr):
    """Decorate the item to skip test unless Clang supports type __attribute__(attr)."""
    def compiler_doesnt_support_struct_attribute(self):
        compiler_path = self.getCompiler()
        f = tempfile.NamedTemporaryFile()
        cmd = [self.getCompiler(), "-x", "c++", "-c", "-o", f.name, "-"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = p.communicate('struct __attribute__((%s)) Test {};'%attr)
        if attr in stderr:
            return "Compiler does not support attribute %s"%(attr)
        return None
    return skipTestIfFn(compiler_doesnt_support_struct_attribute)

def skipUnlessHasCallSiteInfo(func):
    """Decorate the function to skip testing unless call site info from clang is available."""

    def is_compiler_clang_with_call_site_info(self):
        compiler_path = self.getCompiler()
        compiler = os.path.basename(compiler_path)
        if not compiler.startswith("clang"):
            return "Test requires clang as compiler"

        f = tempfile.NamedTemporaryFile()
        cmd = "echo 'int main() {}' | " \
              "%s -g -glldb -O1 -S -emit-llvm -x c -o %s -" % (compiler_path, f.name)
        if os.popen(cmd).close() is not None:
            return "Compiler can't compile with call site info enabled"

        with open(f.name, 'r') as ir_output_file:
            buf = ir_output_file.read()

        if 'DIFlagAllCallsDescribed' not in buf:
            return "Compiler did not introduce DIFlagAllCallsDescribed IR flag"

        return None
    return skipTestIfFn(is_compiler_clang_with_call_site_info)(func)

def skipUnlessThreadSanitizer(func):
    """Decorate the item to skip test unless Clang -fsanitize=thread is supported."""

    def is_compiler_clang_with_thread_sanitizer(self):
        if is_running_under_asan():
            return "Thread sanitizer tests are disabled when runing under ASAN"

        compiler_path = self.getCompiler()
        compiler = os.path.basename(compiler_path)
        if not compiler.startswith("clang"):
            return "Test requires clang as compiler"
        if lldbplatformutil.getPlatform() == 'windows':
            return "TSAN tests not compatible with 'windows'"
        # rdar://28659145 - TSAN tests don't look like they're supported on i386
        if self.getArchitecture() == 'i386' and platform.system() == 'Darwin':
            return "TSAN tests not compatible with i386 targets"
        f = tempfile.NamedTemporaryFile()
        cmd = "echo 'int main() {}' | %s -x c -o %s -" % (compiler_path, f.name)
        if os.popen(cmd).close() is not None:
            return None  # The compiler cannot compile at all, let's *not* skip the test
        cmd = "echo 'int main() {}' | %s -fsanitize=thread -x c -o %s -" % (compiler_path, f.name)
        if os.popen(cmd).close() is not None:
            return "Compiler cannot compile with -fsanitize=thread"
        return None
    return skipTestIfFn(is_compiler_clang_with_thread_sanitizer)(func)

def skipUnlessUndefinedBehaviorSanitizer(func):
    """Decorate the item to skip test unless -fsanitize=undefined is supported."""

    def is_compiler_clang_with_ubsan(self):
        if is_running_under_asan():
            return "Undefined behavior sanitizer tests are disabled when runing under ASAN"

        # Write out a temp file which exhibits UB.
        inputf = tempfile.NamedTemporaryFile(suffix='.c', mode='w')
        inputf.write('int main() { int x = 0; return x / x; }\n')
        inputf.flush()

        # We need to write out the object into a named temp file for inspection.
        outputf = tempfile.NamedTemporaryFile()

        # Try to compile with ubsan turned on.
        cmd = '%s -fsanitize=undefined %s -o %s' % (self.getCompiler(), inputf.name, outputf.name)
        if os.popen(cmd).close() is not None:
            return "Compiler cannot compile with -fsanitize=undefined"

        # Check that we actually see ubsan instrumentation in the binary.
        cmd = 'nm %s' % outputf.name
        with os.popen(cmd) as nm_output:
            if '___ubsan_handle_divrem_overflow' not in nm_output.read():
                return "Division by zero instrumentation is missing"

        # Find the ubsan dylib.
        # FIXME: This check should go away once compiler-rt gains support for __ubsan_on_report.
        cmd = '%s -fsanitize=undefined -x c - -o - -### 2>&1' % self.getCompiler()
        with os.popen(cmd) as cc_output:
            driver_jobs = cc_output.read()
            m = re.search(r'"([^"]+libclang_rt.ubsan_osx_dynamic.dylib)"', driver_jobs)
            if not m:
                return "Could not find the ubsan dylib used by the driver"
            ubsan_dylib = m.group(1)

        # Check that the ubsan dylib has special monitor hooks.
        cmd = 'nm -gU %s' % ubsan_dylib
        with os.popen(cmd) as nm_output:
            syms = nm_output.read()
            if '___ubsan_on_report' not in syms:
                return "Missing ___ubsan_on_report"
            if '___ubsan_get_current_report_data' not in syms:
                return "Missing ___ubsan_get_current_report_data"

        # OK, this dylib + compiler works for us.
        return None

    return skipTestIfFn(is_compiler_clang_with_ubsan)(func)

def is_running_under_asan():
    if ('ASAN_OPTIONS' in os.environ):
        return "ASAN unsupported"
    return None

def skipUnlessAddressSanitizer(func):
    """Decorate the item to skip test unless Clang -fsanitize=thread is supported."""

    def is_compiler_with_address_sanitizer(self):
        # Also don't run tests that use address sanitizer inside an
        # address-sanitized LLDB. The tests don't support that
        # configuration.
        if is_running_under_asan():
            return "Address sanitizer tests are disabled when runing under ASAN"

        compiler_path = self.getCompiler()
        compiler = os.path.basename(compiler_path)
        f = tempfile.NamedTemporaryFile()
        if lldbplatformutil.getPlatform() == 'windows':
            return "ASAN tests not compatible with 'windows'"
        cmd = "echo 'int main() {}' | %s -x c -o %s -" % (compiler_path, f.name)
        if os.popen(cmd).close() is not None:
            return None  # The compiler cannot compile at all, let's *not* skip the test
        cmd = "echo 'int main() {}' | %s -fsanitize=address -x c -o %s -" % (compiler_path, f.name)
        if os.popen(cmd).close() is not None:
            return "Compiler cannot compile with -fsanitize=address"
        return None
    return skipTestIfFn(is_compiler_with_address_sanitizer)(func)

def skipIfAsan(func):
    """Skip this test if the environment is set up to run LLDB *itself* under ASAN."""
    return skipTestIfFn(is_running_under_asan)(func)

def _get_bool_config_skip_if_decorator(key):
    config = lldb.SBDebugger.GetBuildConfiguration()
    value_node = config.GetValueForKey(key)
    fail_value = True # More likely to notice if something goes wrong
    have = value_node.GetValueForKey("value").GetBooleanValue(fail_value)
    return unittest2.skipIf(not have, "requires " + key)

def skipIfCursesSupportMissing(func):
    return _get_bool_config_skip_if_decorator("curses")(func)

def skipIfXmlSupportMissing(func):
    return _get_bool_config_skip_if_decorator("xml")(func)

def skipIfEditlineSupportMissing(func):
    return _get_bool_config_skip_if_decorator("editline")(func)

def skipIfLLVMTargetMissing(target):
    config = lldb.SBDebugger.GetBuildConfiguration()
    targets = config.GetValueForKey("targets").GetValueForKey("value")
    found = False
    for i in range(targets.GetSize()):
        if targets.GetItemAtIndex(i).GetStringValue(99) == target:
            found = True
            break

    return unittest2.skipIf(not found, "requires " + target)

# Call sysctl on darwin to see if a specified hardware feature is available on this machine.
def skipUnlessFeature(feature):
    def is_feature_enabled(self):
        if platform.system() == 'Darwin':
            try:
                DEVNULL = open(os.devnull, 'w')
                output = subprocess.check_output(["/usr/sbin/sysctl", feature], stderr=DEVNULL).decode("utf-8")
                # If 'feature: 1' was output, then this feature is available and
                # the test should not be skipped.
                if re.match('%s: 1\s*' % feature, output):
                    return None
                else:
                    return "%s is not supported on this system." % feature
            except subprocess.CalledProcessError:
                return "%s is not supported on this system." % feature
    return skipTestIfFn(is_feature_enabled)

def skipIfReproducer(func):
    """Skip this test if the environment is set up to run LLDB with reproducers."""
    return unittest2.skipIf(
        configuration.capture_path or configuration.replay_path,
        "reproducers unsupported")(func)
