from __future__ import print_function
from lldbsuite.test import configuration
import re
import subprocess

from builder_base import *

import use_lldb_suite
import lldb

REMOTE_PLATFORM_NAME_RE = re.compile(r"^remote-(.+)$")

def getEffectiveArchitecture(architecture):
    """Return the given architecture if specified, or the value of the architecture in effect."""
    if architecture is not None:
        return architecture
    else:
        return getArchitecture()

def remote_platform_to_triple_os_and_sdk_name(platform_name):
    match = REMOTE_PLATFORM_NAME_RE.match(platform_name)
    if match is None:
        return None, None
    triple_platform = match.group(1)
    if triple_platform == "ios":
        # The iOS SDK does not follow the platform name.
        sdk_name = "iphoneos"
    else:
        # All other SDKs match the platform name.
        sdk_name = triple_platform
    return triple_platform, sdk_name


def construct_triple(platform_name, architecture):
    """Return a fabricated triple for a given platform and architecture."""
    if platform_name is None:
        return None
    elif architecture is None:
        return None

    # Pull the platform name out of the remote platform description.
    triple_platform, sdk_name = remote_platform_to_triple_os_and_sdk_name(
        platform_name)
    if triple_platform is None or sdk_name is None:
        return None

    # Grab the current SDK version number, which will be used in the triple.
    version_output = subprocess.check_output(
        ["xcrun", "--sdk", sdk_name, "--show-sdk-version"]).rstrip().decode('utf-8')
    if version_output is None:
        return None

    return architecture + "-apple-" + triple_platform + version_output


def buildDefault(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries the default way."""
    commands = []
    commands.append(getMake(testdir, testname) + ["all", getArchSpec(architecture),
                     getCCSpec(compiler), getCmdLine(dictionary)])

    triple = construct_triple(
        configuration.lldb_platform_name,
        getEffectiveArchitecture(architecture))
    if triple is not None:
        commands[-1].append("TRIPLE=" + triple)

    runBuildCommands(commands, sender=sender)
    # True signifies that we can handle building default.
    return True


def buildDwarf(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries with dwarf debug info."""
    commands = []
    commands.append(getMake(testdir, testname) +
                    ["MAKE_DSYM=NO", getArchSpec(architecture),
                     getCCSpec(compiler), getCmdLine(dictionary)])

    triple = construct_triple(
        configuration.lldb_platform_name,
        getEffectiveArchitecture(architecture))
    if triple is not None:
        commands[-1].append("TRIPLE=" + triple)

    runBuildCommands(commands, sender=sender)
    # True signifies that we can handle building dwarf.
    return True


def buildGModules(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries with dwarf debug info."""
    commands = []
    commands.append(getMake(testdir, testname) +
                    ["MAKE_DSYM=NO",
                     "MAKE_GMODULES=YES",
                     getArchSpec(architecture),
                     getCCSpec(compiler),
                     "all", getCmdLine(dictionary)])

    triple = construct_triple(
        configuration.lldb_platform_name,
        getEffectiveArchitecture(architecture))
    if triple is not None:
        commands[-1].append("TRIPLE=" + triple)

    runBuildCommands(commands, sender=sender)
    # True signifies that we can handle building with gmodules.
    return True


def buildDsym(
        sender=None,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None):
    """Build the binaries with dsym debug info."""
    commands = []
    commands.append(getMake(testdir, testname) +
                    ["MAKE_DSYM=YES",
                     getArchSpec(architecture),
                     getCCSpec(compiler),
                     "all", getCmdLine(dictionary)])

    triple = construct_triple(
        configuration.lldb_platform_name,
        getEffectiveArchitecture(architecture))
    if triple is not None:
        commands[-1].append("TRIPLE=" + triple)

    runBuildCommands(commands, sender=sender)

    # True signifies that we can handle building dsym.
    return True


