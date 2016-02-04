""" This module contains functions used by the test cases to hide the
architecture and/or the platform dependent nature of the tests. """

from __future__ import absolute_import

# System modules
import re
import subprocess

# Third-party modules
from six.moves.urllib import parse as urlparse

# LLDB modules
from . import configuration
import use_lldb_suite
import lldb

def check_first_register_readable(test_case):
    arch = test_case.getArchitecture()

    if arch in ['x86_64', 'i386']:
        test_case.expect("register read eax", substrs = ['eax = 0x'])
    elif arch in ['arm']:
    	test_case.expect("register read r0", substrs = ['r0 = 0x'])
    elif arch in ['aarch64']:
        test_case.expect("register read x0", substrs = ['x0 = 0x'])
    elif re.match("mips",arch):
        test_case.expect("register read zero", substrs = ['zero = 0x'])
    else:
        # TODO: Add check for other architectures
        test_case.fail("Unsupported architecture for test case (arch: %s)" % test_case.getArchitecture())

def _run_adb_command(cmd, device_id):
    device_id_args = []
    if device_id:
        device_id_args = ["-s", device_id]
    full_cmd = ["adb"] + device_id_args + cmd
    p = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr

def _target_is_android():
    if not hasattr(_target_is_android, 'result'):
        triple = lldb.DBG.GetSelectedPlatform().GetTriple()
        match = re.match(".*-.*-.*-android", triple)
        _target_is_android.result = match is not None
    return _target_is_android.result

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
                ">>> stderr:\n%s\n" % (stdout, stderr))
    return android_device_api.result

def match_android_device(device_arch, valid_archs=None, valid_api_levels=None):
    if not _target_is_android():
        return False
    if valid_archs is not None and device_arch not in valid_archs:
        return False
    if valid_api_levels is not None and android_device_api() not in valid_api_levels:
        return False

    return True

def finalize_build_dictionary(dictionary):
    if _target_is_android():
        if dictionary is None:
            dictionary = {}
        dictionary["OS"] = "Android"
        if android_device_api() >= 16:
            dictionary["PIE"] = 1
    return dictionary
