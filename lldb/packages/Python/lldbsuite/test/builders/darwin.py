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

from lldbsuite.test import configuration


class BuilderDarwin(Builder):
    def getExtraMakeArgs(self):
        """
        Helper function to return extra argumentsfor the make system. This
        method is meant to be overridden by platform specific builders.
        """
        args = dict()

        if configuration.dsymutil:
            args['DSYMUTIL'] = configuration.dsymutil

        operating_system, _ = self.getOsAndEnv()
        if operating_system and operating_system != "macosx":
            builder_dir = os.path.dirname(os.path.abspath(__file__))
            test_dir = os.path.dirname(builder_dir)
            entitlements = os.path.join(test_dir, 'make', 'entitlements.plist')
            args['CODESIGN'] = 'codesign --entitlements {}'.format(entitlements)

        # Return extra args as a formatted string.
        return ' '.join(
            {'{}="{}"'.format(key, value)
             for key, value in args.items()})
    def getOsAndEnv(self):
        if configuration.lldb_platform_name:
            return get_os_env_from_platform(configuration.lldb_platform_name)
        elif configuration.apple_sdk:
            return get_os_from_sdk(configuration.apple_sdk)
        return None, None

    def getArchCFlags(self, architecture):
        """Returns the ARCH_CFLAGS for the make system."""

        # Construct the arch component.
        arch = architecture if architecture else configuration.arch
        if not arch:
            arch = subprocess.check_output(['machine'
                                            ]).rstrip().decode('utf-8')
        if not arch:
            return ""

        # Construct the vendor component.
        vendor = "apple"

        # Construct the os component.
        os, env = self.getOsAndEnv()
        if os is None or env is None:
            return ""

        # Get the SDK from the os and env.
        sdk = lldbutil.get_xcode_sdk(os, env)
        if not sdk:
            return ""

        version = lldbutil.get_xcode_sdk_version(sdk)
        if not version:
            return ""

        # Construct the triple from its components.
        triple = "{}-{}-{}-{}".format(vendor, os, version, env)

        # Construct min version argument
        version_min = ""
        if env == "simulator":
            version_min = "-m{}-simulator-version-min={}".format(os, version)
        elif os == "macosx":
            version_min = "-m{}-version-min={}".format(os, version)

        return "ARCH_CFLAGS=\"-target {} {}\"".format(triple, version_min)

    def buildDsym(self,
                  sender=None,
                  architecture=None,
                  compiler=None,
                  dictionary=None,
                  testdir=None,
                  testname=None):
        """Build the binaries with dsym debug info."""
        commands = []
        commands.append(
            self.getMake(testdir, testname) + [
                "MAKE_DSYM=YES",
                self.getArchCFlags(architecture),
                self.getArchSpec(architecture),
                self.getCCSpec(compiler),
                self.getExtraMakeArgs(),
                self.getSDKRootSpec(),
                self.getModuleCacheSpec(), "all",
                self.getCmdLine(dictionary)
            ])

        self.runBuildCommands(commands, sender=sender)

        # True signifies that we can handle building dsym.
        return True
