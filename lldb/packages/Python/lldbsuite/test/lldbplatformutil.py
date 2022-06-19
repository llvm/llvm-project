""" This module contains functions used by the test cases to hide the
architecture and/or the platform dependent nature of the tests. """

from __future__ import absolute_import

# System modules
import ctypes
import itertools
import os
import re
import subprocess
import sys

# Third-party modules
import six
from six.moves.urllib import parse as urlparse

# LLDB modules
from . import configuration
import lldb
import lldbsuite.test.lldbplatform as lldbplatform


def check_first_register_readable(test_case):
    arch = test_case.getArchitecture()

    if arch in ['x86_64', 'i386']:
        test_case.expect("register read eax", substrs=['eax = 0x'])
    elif arch in ['arm', 'armv7', 'armv7k', 'armv8l', 'armv7l']:
        test_case.expect("register read r0", substrs=['r0 = 0x'])
    elif arch in ['aarch64', 'arm64', 'arm64e', 'arm64_32']:
        test_case.expect("register read x0", substrs=['x0 = 0x'])
    elif re.match("mips", arch):
        test_case.expect("register read zero", substrs=['zero = 0x'])
    elif arch in ['s390x']:
        test_case.expect("register read r0", substrs=['r0 = 0x'])
    elif arch in ['powerpc64le']:
        test_case.expect("register read r0", substrs=['r0 = 0x'])
    else:
        # TODO: Add check for other architectures
        test_case.fail(
            "Unsupported architecture for test case (arch: %s)" %
            test_case.getArchitecture())


def _run_adb_command(cmd, device_id):
    device_id_args = []
    if device_id:
        device_id_args = ["-s", device_id]
    full_cmd = ["adb"] + device_id_args + cmd
    p = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr


def target_is_android():
    return configuration.lldb_platform_name == "remote-android"

def android_device_api():
    if not hasattr(android_device_api, 'result'):
        assert configuration.lldb_platform_url is not None
        device_id = None
        parsed_url = urlparse.urlparse(configuration.lldb_platform_url)
        host_name = parsed_url.netloc.split(":")[0]
        if host_name != 'localhost':
            device_id = host_name
            if device_id.startswith('[') and device_id.endswith(']'):
                device_id = device_id[1:-1]
        retcode, stdout, stderr = _run_adb_command(
            ["shell", "getprop", "ro.build.version.sdk"], device_id)
        if retcode == 0:
            android_device_api.result = int(stdout)
        else:
            raise LookupError(
                ">>> Unable to determine the API level of the Android device.\n"
                ">>> stdout:\n%s\n"
                ">>> stderr:\n%s\n" %
                (stdout, stderr))
    return android_device_api.result


def match_android_device(device_arch, valid_archs=None, valid_api_levels=None):
    if not target_is_android():
        return False
    if valid_archs is not None and device_arch not in valid_archs:
        return False
    if valid_api_levels is not None and android_device_api() not in valid_api_levels:
        return False

    return True


def finalize_build_dictionary(dictionary):
    if target_is_android():
        if dictionary is None:
            dictionary = {}
        dictionary["OS"] = "Android"
        dictionary["PIE"] = 1
    return dictionary


def getHostPlatform():
    """Returns the host platform running the test suite."""
    # Attempts to return a platform name matching a target Triple platform.
    if sys.platform.startswith('linux'):
        return 'linux'
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        return 'windows'
    elif sys.platform.startswith('darwin'):
        return 'macosx'
    elif sys.platform.startswith('freebsd'):
        return 'freebsd'
    elif sys.platform.startswith('netbsd'):
        return 'netbsd'
    else:
        return sys.platform


def getDarwinOSTriples():
    return lldbplatform.translate(lldbplatform.darwin_all)

def getPlatform():
    """Returns the target platform which the tests are running on."""
    # Use the Apple SDK to determine the platform if set.
    if configuration.apple_sdk:
        platform = configuration.apple_sdk
        dot = platform.find('.')
        if dot != -1:
            platform = platform[:dot]
        if platform == 'iphoneos':
            platform = 'ios'
        return platform

    platform = configuration.lldb_platform_name
    if platform is None:
        platform = "host"
    if platform == "qemu-user":
        platform = "host"
    if platform == "host":
        return getHostPlatform()
    if platform.startswith("remote-"):
        return platform[7:]
    return platform


def platformIsDarwin():
    """Returns true if the OS triple for the selected platform is any valid apple OS"""
    return getPlatform() in getDarwinOSTriples()


def findMainThreadCheckerDylib():
    if not platformIsDarwin():
        return ""

    if getPlatform() in lldbplatform.translate(lldbplatform.darwin_embedded):
        return "/Developer/usr/lib/libMainThreadChecker.dylib"

    with os.popen('xcode-select -p') as output:
        xcode_developer_path = output.read().strip()
        mtc_dylib_path = '%s/usr/lib/libMainThreadChecker.dylib' % xcode_developer_path
        if os.path.isfile(mtc_dylib_path):
            return mtc_dylib_path

    return ""


class _PlatformContext(object):
    """Value object class which contains platform-specific options."""

    def __init__(self, shlib_environment_var, shlib_path_separator, shlib_prefix, shlib_extension):
        self.shlib_environment_var = shlib_environment_var
        self.shlib_path_separator = shlib_path_separator
        self.shlib_prefix = shlib_prefix
        self.shlib_extension = shlib_extension


def createPlatformContext():
    if platformIsDarwin():
        return _PlatformContext('DYLD_LIBRARY_PATH', ':', 'lib', 'dylib')
    elif getPlatform() in ("freebsd", "linux", "netbsd"):
        return _PlatformContext('LD_LIBRARY_PATH', ':', 'lib', 'so')
    else:
        return _PlatformContext('PATH', ';', '', 'dll')


def hasChattyStderr(test_case):
    """Some targets produce garbage on the standard error output. This utility function
    determines whether the tests can be strict about the expected stderr contents."""
    if match_android_device(test_case.getArchitecture(), ['aarch64'], range(22, 25+1)):
        return True  # The dynamic linker on the device will complain about unknown DT entries
    return False

if getHostPlatform() == "linux":
    def enable_attach():
        """Enable attaching to _this_ process, if host requires such an action.
        Suitable for use as a preexec_fn in subprocess.Popen and similar."""
        c = ctypes.CDLL(None)
        PR_SET_PTRACER = ctypes.c_int(0x59616d61)
        PR_SET_PTRACER_ANY = ctypes.c_ulong(-1)
        c.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)
else:
    enable_attach = None
