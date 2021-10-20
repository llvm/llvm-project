import re
import os
import subprocess

from .builder import Builder
from lldbsuite.test import configuration
import lldbsuite.test.lldbutil as lldbutil

REMOTE_PLATFORM_NAME_RE = re.compile(r"^remote-(.+)$")
SIMULATOR_PLATFORM_RE = re.compile(r"^(.+)-simulator$")


def get_os_env_from_platform(platform):
    match = REMOTE_PLATFORM_NAME_RE.match(platform)
    if match:
        return match.group(1), ""
    match = SIMULATOR_PLATFORM_RE.match(platform)
    if match:
        return match.group(1), "simulator"
    return None, None


def get_os_from_sdk(sdk):
    return sdk[:sdk.find('.')], ""


def get_os_and_env():
    if configuration.lldb_platform_name:
        return get_os_env_from_platform(configuration.lldb_platform_name)
    if configuration.apple_sdk:
        return get_os_from_sdk(configuration.apple_sdk)
    return None, None


def get_triple():
    # Construct the vendor component.
    vendor = "apple"

    # Construct the os component.
    os, env = get_os_and_env()
    if os is None or env is None:
        return None, None, None, None

    # Get the SDK from the os and env.
    sdk = lldbutil.get_xcode_sdk(os, env)
    if sdk is None:
        return None, None, None, None

    # Get the version from the SDK.
    version = lldbutil.get_xcode_sdk_version(sdk)
    if version is None:
        return None, None, None, None

    return vendor, os, version, env


class BuilderDarwin(Builder):
    def getExtraMakeArgs(self):
        """
        Helper function to return extra argumentsfor the make system. This
        method is meant to be overridden by platform specific builders.
        """
        args = dict()

        if configuration.dsymutil:
            args['DSYMUTIL'] = configuration.dsymutil

        operating_system, env = get_os_and_env()
        if operating_system and operating_system != "macosx":
            builder_dir = os.path.dirname(os.path.abspath(__file__))
            test_dir = os.path.dirname(builder_dir)
            if env == "simulator":
              entitlements_file = 'entitlements-simulator.plist'
            else:
              entitlements_file = 'entitlements.plist'
            entitlements = os.path.join(test_dir, 'make', entitlements_file)
            args['CODESIGN'] = 'codesign --entitlements {}'.format(
                entitlements)
        else:
            args['CODESIGN'] = 'codesign'

        # Return extra args as a formatted string.
        return ['{}={}'.format(key, value) for key, value in args.items()]

    def getArchCFlags(self, arch):
        """Returns the ARCH_CFLAGS for the make system."""
        # Get the triple components.
        vendor, os, version, env = get_triple()
        if vendor is None or os is None or version is None or env is None:
            return []

        # Construct the triple from its components.
        triple = '-'.join([arch, vendor, os, version, env])

        # Construct min version argument
        version_min = ""
        if env == "simulator":
            version_min = "-m{}-simulator-version-min={}".format(os, version)
        elif os == "macosx":
            version_min = "-m{}-version-min={}".format(os, version)

        return ["ARCH_CFLAGS=-target {} {}".format(triple, version_min)]

    def _getDebugInfoArgs(self, debug_info):
        if debug_info == "dsym":
            return ["MAKE_DSYM=YES"]
        return super(BuilderDarwin, self)._getDebugInfoArgs(debug_info)
