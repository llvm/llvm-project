#!/usr/bin/env python

import argparse
import os
import os.path
import shutil
import subprocess
import sys


class BuildError(Exception):

    def __init__(self,
                 string=None,
                 path=None,
                 inferior_error=None):
        self.m_string = string
        self.m_path = path
        self.m_inferior_error = inferior_error

    def __str__(self):
        if self.m_path and self.m_string:
            return "Build error: %s (referring to %s)" % (
                self.m_string, self.m_path)
        if self.m_path:
            return "Build error (referring to %s)" % (self.m_path)
        if self.m_string:
            return "Build error: %s" % (self.m_string)
        return "Build error"


class LLDBBuildBot:

    def __init__(
            self,
            build_directory_path,
            log_path,
            lldb_repository_url="http://llvm.org/svn/llvm-project/lldb/trunk",
            llvm_repository_url="http://llvm.org/svn/llvm-project/llvm/trunk",
            clang_repository_url="http://llvm.org/svn/llvm-project/cfe/trunk",
            revision=None):
        self.m_build_directory_path = os.path.abspath(build_directory_path)
        self.m_log_path = os.path.abspath(log_path)
        self.m_lldb_repository_url = lldb_repository_url
        self.m_llvm_repository_url = llvm_repository_url
        self.m_clang_repository_url = clang_repository_url
        self.m_revision = revision
        self.m_log_stream = None

    def Setup(self):
        if os.path.exists(self.m_build_directory_path):
            raise BuildError(
                string="Build directory exists",
                path=self.m_build_directory_path)
        if os.path.exists(self.m_log_path):
            raise BuildError(string="Log file exists", path=self.m_log_path)
        self.m_log_stream = open(self.m_log_path, 'w')
        os.mkdir(self.m_build_directory_path)

    def Checkout(self):
        os.chdir(self.m_build_directory_path)

        cmdline_prefix = []

        if self.m_revision is not None:
            cmdline_prefix = ["svn", "-r %s" % (self.m_revision), "co"]
        else:
            cmdline_prefix = ["svn", "co"]

        returncode = subprocess.call(
            cmdline_prefix + [
                self.m_lldb_repository_url,
                "lldb"],
            stdout=self.m_log_stream,
            stderr=self.m_log_stream)
        if returncode != 0:
            raise BuildError(string="Couldn't checkout LLDB")

        os.chdir("lldb")

        returncode = subprocess.call(
            cmdline_prefix + [
                self.m_llvm_repository_url,
                "llvm.checkout"],
            stdout=self.m_log_stream,
            stderr=self.m_log_stream)

        if returncode != 0:
            raise BuildError(string="Couldn't checkout LLVM")

        os.symlink("llvm.checkout", "llvm")

        os.chdir("llvm/tools")

        returncode = subprocess.call(
            cmdline_prefix + [
                self.m_clang_repository_url,
                "clang"],
            stdout=self.m_log_stream,
            stderr=self.m_log_stream)

        if returncode != 0:
            raise BuildError(string="Couldn't checkout Clang")

    def Build(self):
        os.chdir(self.m_build_directory_path)
        os.chdir("lldb/llvm")

        returncode = subprocess.call(["./configure",
                                      "--disable-optimized",
                                      "--enable-assertions",
                                      "--enable-targets=x86,x86_64,arm"],
                                     stdout=self.m_log_stream,
                                     stderr=self.m_log_stream)

        if returncode != 0:
            raise BuildError(string="Couldn't configure LLVM/Clang")

        returncode = subprocess.call(["make"],
                                     stdout=self.m_log_stream,
                                     stderr=self.m_log_stream)

        if returncode != 0:
            raise BuildError(string="Couldn't build LLVM/Clang")

        os.chdir(self.m_build_directory_path)
        os.chdir("lldb")

        returncode = subprocess.call(["xcodebuild",
                                      "-project", "lldb.xcodeproj",
                                      "-target", "lldb-tool",
                                      "-configuration", "Debug",
                                      "-arch", "x86_64",
                                      "LLVM_CONFIGURATION=Debug+Asserts",
                                      "OBJROOT=build"],
                                     stdout=self.m_log_stream,
                                     stderr=self.m_log_stream)

        if returncode != 0:
            raise BuildError(string="Couldn't build LLDB")

    def Test(self):
        os.chdir(self.m_build_directory_path)
        os.chdir("lldb/test")

        returncode = subprocess.call(["./dotest.py", "-t"],
                                     stdout=self.m_log_stream,
                                     stderr=self.m_log_stream)

    def Takedown(self):
        os.chdir("/tmp")
        self.m_log_stream.close()
        shutil.rmtree(self.m_build_directory_path)

    def Run(self):
        self.Setup()
        self.Checkout()
        self.Build()
        # self.Test()
        self.Takedown()


def GetArgParser():
    parser = argparse.ArgumentParser(
        description="Try to build LLDB/LLVM/Clang and run the full test suite.")
    parser.add_argument(
        "--build-path",
        "-b",
        required=True,
        help="A (nonexistent) path to put temporary build products into",
        metavar="path")
    parser.add_argument(
        "--log-file",
        "-l",
        required=True,
        help="The name of a (nonexistent) log file",
        metavar="file")
    parser.add_argument(
        "--revision",
        "-r",
        required=False,
        help="The LLVM revision to use",
        metavar="N")
    return parser

parser = GetArgParser()
arg_dict = vars(parser.parse_args())

build_bot = LLDBBuildBot(build_directory_path=arg_dict["build_path"],
                         log_path=arg_dict["log_file"],
                         revision=arg_dict["revision"])

try:
    build_bot.Run()
except BuildError as err:
    print(err)
