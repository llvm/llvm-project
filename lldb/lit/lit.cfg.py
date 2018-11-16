# -*- Python -*-

import os
import sys
import re
import platform
import shutil
import subprocess

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = 'LLDB'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = ['.test', '.cpp', '.s']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.lldb_obj_root, 'lit')

# Tweak the PATH to include the tools dir.
llvm_config.with_system_environment('PATH')
llvm_config.with_environment('PATH', config.lldb_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

llvm_config.with_environment('LD_LIBRARY_PATH', config.lldb_libs_dir, append_path=True)
llvm_config.with_environment('LD_LIBRARY_PATH', config.llvm_libs_dir, append_path=True)
llvm_config.with_system_environment('LD_LIBRARY_PATH', append_path=True)


llvm_config.use_default_substitutions()

if platform.system() in ['Darwin']:
    debugserver = lit.util.which('debugserver', config.lldb_tools_dir)
else:
    debugserver = lit.util.which('lldb-server', config.lldb_tools_dir)
lldb = "%s -S %s/lit-lldb-init" % (lit.util.which('lldb', config.lldb_tools_dir),
                               config.test_source_root)

lldbmi = lit.util.which('lldb-mi', config.lldb_tools_dir)
if lldbmi:
    config.available_features.add('lldb-mi')

config.cc = llvm_config.use_llvm_tool(config.cc, required=True)
config.cxx = llvm_config.use_llvm_tool(config.cxx, required=True)

if platform.system() in ['Darwin']:
    try:
        out = subprocess.check_output(['xcrun', '--show-sdk-path']).strip()
        res = 0
    except OSError:
        res = -1
    if res == 0 and out:
        sdk_path = lit.util.to_string(out)
        lit_config.note('using SDKROOT: %r' % sdk_path)
        config.cc += " -isysroot %s" % sdk_path
        config.cxx += " -isysroot %s" % sdk_path

if platform.system() in ['OpenBSD']:
    config.cc += " -pthread"
    config.cxx += " -pthread"

config.substitutions.append(('%cc', config.cc))
config.substitutions.append(('%cxx', config.cxx))

if lldbmi:
  config.substitutions.append(('%lldbmi', lldbmi + " --synchronous"))
config.substitutions.append(('%lldb', lldb))

if debugserver is not None:
    if platform.system() in ['Darwin']:
        config.substitutions.append(('%debugserver', debugserver))
    else:
        config.substitutions.append(('%debugserver', debugserver + ' gdbserver'))

tools = ['lldb-test', 'yaml2obj', 'obj2yaml', 'llvm-pdbutil']
llvm_config.add_tool_substitutions(tools, [config.llvm_tools_dir, config.lldb_tools_dir])

if re.match(r'^arm(hf.*-linux)|(.*-linux-gnuabihf)', config.target_triple):
    config.available_features.add("armhf-linux")

print("config.cc = {}".format(config.cc))
if re.match(r'icc', config.cc):
    config.available_features.add("compiler-icc")
elif re.match(r'clang', config.cc):
    config.available_features.add("compiler-clang")
elif re.match(r'gcc', config.cc):
    config.available_features.add("compiler-gcc")
elif re.match(r'cl', config.cc):
    config.available_features.add("compiler-msvc")

if config.have_lld:
  config.available_features.add("lld")

def calculate_arch_features(arch_string):
    # This will add a feature such as x86, arm, mips, etc for each built
    # target
    features = []
    for arch in arch_string.split():
        features.append(arch.lower())
    return features

# Run llvm-config and add automatically add features for whether we have
# assertions enabled, whether we are in debug mode, and what targets we
# are built for.
llvm_config.feature_config(
    [('--assertion-mode', {'ON': 'asserts'}),
     ('--build-mode', {'DEBUG': 'debug'}),
     ('--targets-built', calculate_arch_features)
     ])

# Clean the module caches in the test build directory.  This is
# necessary in an incremental build whenever clang changes underneath,
# so doing it once per lit.py invocation is close enough.
for i in ['module-cache-clang', 'module-cache-lldb']:
    cachedir = os.path.join(config.llvm_obj_root, 'lldb-test-build.noindex', i)
    if os.path.isdir(cachedir):
        print("Deleting module cache at %s."%cachedir)
        shutil.rmtree(cachedir)
