import os
import pathlib
import platform
import subprocess
import sys
import itertools

import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbplatformutil as lldbplatformutil
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test import configuration
from lldbsuite.test_event import build_exception
from lldbsuite.support import seven


class Builder:
    def getArchitecture(self):
        """Returns the architecture in effect the test suite is running with."""
        return configuration.arch if configuration.arch else ""

    def getCompiler(self):
        """Returns the compiler in effect the test suite is running with."""
        compiler = configuration.compiler if configuration.compiler else "clang"
        compiler = lldbutil.which(compiler)
        return os.path.abspath(compiler)

    def getTriple(self, arch):
        """Returns the triple for the given architecture or None."""
        return None

    def getExtraMakeArgs(self):
        """
        Helper function to return extra argumentsfor the make system. This
        method is meant to be overridden by platform specific builders.
        """
        return []

    def getArchCFlags(self, architecture):
        """Returns the ARCH_CFLAGS for the make system."""
        return []

    def getMake(self, test_subdir, test_name):
        """Returns the invocation for GNU make.
        The first argument is a tuple of the relative path to the testcase
        and its filename stem."""
        # Construct the base make invocation.
        lldb_test = os.environ["LLDB_TEST"]
        if not (
            lldb_test
            and configuration.test_build_dir
            and test_subdir
            and test_name
            and (not os.path.isabs(test_subdir))
        ):
            raise Exception("Could not derive test directories")
        build_dir = os.path.join(configuration.test_build_dir, test_subdir, test_name)
        src_dir = os.path.join(configuration.test_src_root, test_subdir)
        # This is a bit of a hack to make inline testcases work.
        makefile = os.path.join(src_dir, "Makefile")
        if not os.path.isfile(makefile):
            makefile = os.path.join(build_dir, "Makefile")
        return [
            configuration.make_path,
            "VPATH=" + src_dir,
            "-C",
            build_dir,
            "-I",
            src_dir,
            "-I",
            os.path.join(lldb_test, "make"),
            "-f",
            makefile,
        ]

    def getCmdLine(self, d):
        """
        Helper function to return a command line argument string used for the
        make system.
        """

        # If d is None or an empty mapping, just return an empty list.
        if not d:
            return []

        def setOrAppendVariable(k, v):
            append_vars = ["CFLAGS", "CFLAGS_EXTRAS", "LD_EXTRAS"]
            if k in append_vars and k in os.environ:
                v = os.environ[k] + " " + v
            return "%s=%s" % (k, v)

        cmdline = [setOrAppendVariable(k, v) for k, v in list(d.items())]

        return cmdline

    def getArchSpec(self, architecture):
        """
        Helper function to return the key-value string to specify the architecture
        used for the make system.
        """
        return ["ARCH=" + architecture] if architecture else []

    def getToolchainSpec(self, compiler):
        """
        Helper function to return the key-value strings to specify the toolchain
        used for the make system.
        """
        cc = compiler if compiler else None
        if not cc and configuration.compiler:
            cc = configuration.compiler

        if not cc:
            return []

        exe_ext = ""
        if lldbplatformutil.getHostPlatform() == "windows":
            exe_ext = ".exe"

        cc = cc.strip()
        cc_path = pathlib.Path(cc)

        # We can get CC compiler string in the following formats:
        #  [<tool>] <compiler>    - such as 'xrun clang', 'xrun /usr/bin/clang' & etc
        #
        # Where <compiler> could contain the following parts:
        #   <simple-name>[.<exe-ext>]                           - sucn as 'clang', 'clang.exe' ('clang-cl.exe'?)
        #   <target-triple>-<simple-name>[.<exe-ext>]           - such as 'armv7-linux-gnueabi-gcc'
        #   <path>/<simple-name>[.<exe-ext>]                    - such as '/usr/bin/clang', 'c:\path\to\compiler\clang,exe'
        #   <path>/<target-triple>-<simple-name>[.<exe-ext>]    - such as '/usr/bin/clang', 'c:\path\to\compiler\clang,exe'

        cc_ext = cc_path.suffix
        # Compiler name without extension
        cc_name = cc_path.stem.split(" ")[-1]

        # A kind of compiler (canonical name): clang, gcc, cc & etc.
        cc_type = cc_name
        # A triple prefix of compiler name: <armv7-none-linux-gnu->gcc
        cc_prefix = ""
        if not "clang-cl" in cc_name and not "llvm-gcc" in cc_name:
            cc_name_parts = cc_name.split("-")
            cc_type = cc_name_parts[-1]
            if len(cc_name_parts) > 1:
                cc_prefix = "-".join(cc_name_parts[:-1]) + "-"

        # A kind of C++ compiler.
        cxx_types = {
            "icc": "icpc",
            "llvm-gcc": "llvm-g++",
            "gcc": "g++",
            "cc": "c++",
            "clang": "clang++",
        }
        cxx_type = cxx_types.get(cc_type, cc_type)

        cc_dir = cc_path.parent

        def getToolchainUtil(util_name):
            return os.path.join(configuration.llvm_tools_dir, util_name + exe_ext)

        cxx = cc_dir / (cc_prefix + cxx_type + cc_ext)

        util_names = {
            "OBJCOPY": "objcopy",
            "STRIP": "strip",
            "ARCHIVER": "ar",
            "DWP": "dwp",
        }
        utils = []

        # Required by API TestBSDArchives.py tests.
        if not os.getenv("LLVM_AR"):
            utils.extend(["LLVM_AR=%s" % getToolchainUtil("llvm-ar")])

        if cc_type in ["clang", "cc", "gcc"]:
            util_paths = {}
            # Assembly a toolchain side tool cmd based on passed CC.
            for var, name in util_names.items():
                # Do not override explicity specified tool from the cmd line.
                if not os.getenv(var):
                    util_paths[var] = getToolchainUtil("llvm-" + name)
                else:
                    util_paths[var] = os.getenv(var)
            utils.extend(["AR=%s" % util_paths["ARCHIVER"]])

            # Look for llvm-dwp or gnu dwp
            if not lldbutil.which(util_paths["DWP"]):
                util_paths["DWP"] = getToolchainUtil("llvm-dwp")
            if not lldbutil.which(util_paths["DWP"]):
                util_paths["DWP"] = lldbutil.which("llvm-dwp")
            if not util_paths["DWP"]:
                util_paths["DWP"] = lldbutil.which("dwp")
                if not util_paths["DWP"]:
                    del util_paths["DWP"]

            if lldbplatformutil.platformIsDarwin():
                util_paths["STRIP"] = seven.get_command_output("xcrun -f strip")

            for var, path in util_paths.items():
                utils.append("%s=%s" % (var, path))

        if lldbplatformutil.platformIsDarwin():
            utils.extend(["AR=%slibtool" % os.getenv("CROSS_COMPILE", "")])

        return [
            "CC=%s" % cc,
            "CC_TYPE=%s" % cc_type,
            "CXX=%s" % cxx,
        ] + utils

    def getSDKRootSpec(self):
        """
        Helper function to return the key-value string to specify the SDK root
        used for the make system.
        """
        if configuration.sdkroot:
            return ["SDKROOT={}".format(configuration.sdkroot)]
        return []

    def getModuleCacheSpec(self):
        """
        Helper function to return the key-value string to specify the clang
        module cache used for the make system.
        """
        if configuration.clang_module_cache_dir:
            return [
                "CLANG_MODULE_CACHE_DIR={}".format(configuration.clang_module_cache_dir)
            ]
        return []

    def getLibCxxArgs(self):
        if configuration.libcxx_include_dir and configuration.libcxx_library_dir:
            libcpp_args = [
                "LIBCPP_INCLUDE_DIR={}".format(configuration.libcxx_include_dir),
                "LIBCPP_LIBRARY_DIR={}".format(configuration.libcxx_library_dir),
            ]
            if configuration.libcxx_include_target_dir:
                libcpp_args.append(
                    "LIBCPP_INCLUDE_TARGET_DIR={}".format(
                        configuration.libcxx_include_target_dir
                    )
                )
            return libcpp_args
        return []

    def getLLDBObjRoot(self):
        return ["LLDB_OBJ_ROOT={}".format(configuration.lldb_obj_root)]

    def _getDebugInfoArgs(self, debug_info):
        if debug_info is None:
            return []
        if debug_info == "dwarf":
            return ["MAKE_DSYM=NO"]
        if debug_info == "dwo":
            return ["MAKE_DSYM=NO", "MAKE_DWO=YES"]
        if debug_info == "gmodules":
            return ["MAKE_DSYM=NO", "MAKE_GMODULES=YES"]
        return None

    def getBuildCommand(
        self,
        debug_info,
        architecture=None,
        compiler=None,
        dictionary=None,
        testdir=None,
        testname=None,
        make_targets=None,
    ):
        debug_info_args = self._getDebugInfoArgs(debug_info)
        if debug_info_args is None:
            return None
        if make_targets is None:
            make_targets = ["all"]
        command_parts = [
            self.getMake(testdir, testname),
            debug_info_args,
            make_targets,
            self.getArchCFlags(architecture),
            self.getArchSpec(architecture),
            self.getToolchainSpec(compiler),
            self.getExtraMakeArgs(),
            self.getSDKRootSpec(),
            self.getModuleCacheSpec(),
            self.getLibCxxArgs(),
            self.getLLDBObjRoot(),
            self.getCmdLine(dictionary),
        ]
        command = list(itertools.chain(*command_parts))

        return command

    def cleanup(self, dictionary=None):
        """Perform a platform-specific cleanup after the test."""
        return True
