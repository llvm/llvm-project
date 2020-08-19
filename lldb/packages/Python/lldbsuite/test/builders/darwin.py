import re
import subprocess

from .builder import Builder
from lldbsuite.test import configuration

REMOTE_PLATFORM_NAME_RE = re.compile(r"^remote-(.+)$")
SIMULATOR_PLATFORM_RE = re.compile(r"^(.+)-simulator$")


def get_sdk(os, env):
    if os == "ios":
        if env == "simulator":
            return "iphonesimulator"
        if env == "macabi":
            return "macosx"
        return "iphoneos"
    elif os == "tvos":
        if env == "simulator":
            return "appletvsimulator"
        return "appletvos"
    elif os == "watchos":
        if env == "simulator":
            return "watchsimulator"
        return "watchos"
    return os


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

        os, _ = self.getOsAndEnv()
        if os and os != "macosx":
            args['CODESIGN'] = 'codesign'

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
        sdk = get_sdk(os, env)
        if not sdk:
            return ""

        version = subprocess.check_output(
            ["xcrun", "--sdk", sdk,
             "--show-sdk-version"]).rstrip().decode('utf-8')
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
