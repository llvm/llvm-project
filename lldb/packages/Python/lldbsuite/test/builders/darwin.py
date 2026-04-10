import re
import os
import subprocess

from .builder import Builder
from lldbsuite.test import configuration
import lldbsuite.test.lldbutil as lldbutil

TRIPLE_RE = re.compile(
    r"^(?P<arch>[a-zA-Z0-9_]+)"  # arch (required)
    r"(?:-(?P<vendor>[a-zA-Z0-9_]+))?"  # vendor (optional)
    r"(?:-(?P<os>[a-zA-Z_]+)(?P<os_version>[\d.]+)?)?"  # os + version (optional)
    r"(?:-(?P<env>[a-zA-Z0-9_]+))?"  # env/abi (optional)
    r"$"
)


def split_triple(triple):
    m = TRIPLE_RE.match(triple)
    if m:
        return m["arch"], m["vendor"], m["os"], m["os_version"], m["env"]
    return None, None, None, None, None


class BuilderDarwin(Builder):
    def getExtraMakeArgs(self):
        """
        Helper function to return extra argumentsfor the make system. This
        method is meant to be overridden by platform specific builders.
        """
        args = dict()

        if configuration.dsymutil:
            args["DSYMUTIL"] = configuration.dsymutil

        if configuration.apple_sdk and "internal" in configuration.apple_sdk:
            sdk_root = lldbutil.get_xcode_sdk_root(configuration.apple_sdk)
            if sdk_root:
                private_frameworks = os.path.join(
                    sdk_root, "System", "Library", "PrivateFrameworks"
                )
                args["FRAMEWORK_INCLUDES"] = "-F{}".format(private_frameworks)

        triple = self.getTriple()
        if triple:
            _, _, operating_system, _, env = split_triple(triple)

            builder_dir = os.path.dirname(os.path.abspath(__file__))
            test_dir = os.path.dirname(builder_dir)
            if operating_system in [None, "darwin", "macos", "macosx"]:
                entitlements_file = "entitlements-macos.plist"
            else:
                if env == "simulator":
                    entitlements_file = "entitlements-simulator.plist"
                else:
                    entitlements_file = "entitlements.plist"
            entitlements = os.path.join(test_dir, "make", entitlements_file)
            args["CODESIGN"] = "codesign --entitlements {}".format(entitlements)

        # Return extra args as a formatted string.
        return ["{}={}".format(key, value) for key, value in args.items()]

    def getArchCFlags(self):
        """Returns the ARCH_CFLAGS for the make system."""
        # Get the triple components.
        triple = self.getTriple()
        if not triple:
            return []

        _, _, os, version, _ = split_triple(triple)

        if os == "darwin" or not version:
            return []

        target_os = "-mtargetos={}{}".format(os, version)

        return ["ARCH_CFLAGS={}".format(target_os)]

    def _getDebugInfoArgs(self, debug_info):
        if debug_info == "dsym":
            return ["MAKE_DSYM=YES"]
        return super(BuilderDarwin, self)._getDebugInfoArgs(debug_info)
