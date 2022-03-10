# -*- Python -*-

import os
import platform
import re
import shlex
import subprocess

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Clang Tools'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.hpp', '.m', '.mm', '.cu', '.ll', '.cl', '.s',
  '.modularize', '.module-map-checker', '.test']

# Test-time dependencies located in directories called 'Inputs' are excluded
# from test suites; there won't be any lit tests within them.
config.excludes = ['Inputs']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.clang_tools_binary_dir, 'test')

# Clear some environment variables that might affect Clang.
#
# This first set of vars are read by Clang, but shouldn't affect tests
# that aren't specifically looking for these features, or are required
# simply to run the tests at all.
#
# FIXME: Should we have a tool that enforces this?

# safe_env_vars = ('TMPDIR', 'TEMP', 'TMP', 'USERPROFILE', 'PWD',
#                  'MACOSX_DEPLOYMENT_TARGET', 'IPHONEOS_DEPLOYMENT_TARGET',
#                  'IOS_SIMULATOR_DEPLOYMENT_TARGET',
#                  'VCINSTALLDIR', 'VC100COMNTOOLS', 'VC90COMNTOOLS',
#                  'VC80COMNTOOLS')
possibly_dangerous_env_vars = ['COMPILER_PATH', 'RC_DEBUG_OPTIONS',
                               'CINDEXTEST_PREAMBLE_FILE', 'LIBRARY_PATH',
                               'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH',
                               'OBJC_INCLUDE_PATH', 'OBJCPLUS_INCLUDE_PATH',
                               'LIBCLANG_TIMING', 'LIBCLANG_OBJTRACKING',
                               'LIBCLANG_LOGGING', 'LIBCLANG_BGPRIO_INDEX',
                               'LIBCLANG_BGPRIO_EDIT', 'LIBCLANG_NOTHREADS',
                               'LIBCLANG_RESOURCE_USAGE',
                               'LIBCLANG_CODE_COMPLETION_LOGGING']
# Clang/Win32 may refer to %INCLUDE%. vsvarsall.bat sets it.
if platform.system() != 'Windows':
    possibly_dangerous_env_vars.append('INCLUDE')
for name in possibly_dangerous_env_vars:
  if name in config.environment:
    del config.environment[name]

# Tweak the PATH to include the tools dir and the scripts dir.
path = os.path.pathsep.join((
        config.clang_tools_dir, config.llvm_tools_dir, config.environment['PATH']))
config.environment['PATH'] = path

path = os.path.pathsep.join((config.clang_libs_dir, config.llvm_libs_dir,
                              config.environment.get('LD_LIBRARY_PATH','')))
config.environment['LD_LIBRARY_PATH'] = path

if config.clang_tidy_staticanalyzer:
    config.available_features.add('static-analyzer')

python_exec = shlex.quote(config.python_executable)
check_clang_tidy = os.path.join(
    config.test_source_root, "clang-tidy", "check_clang_tidy.py")
config.substitutions.append(
    ('%check_clang_tidy',
     '%s %s' % (python_exec, check_clang_tidy)) )
clang_tidy_diff = os.path.join(
    config.test_source_root, "..", "clang-tidy", "tool", "clang-tidy-diff.py")
config.substitutions.append(
    ('%clang_tidy_diff',
     '%s %s' % (python_exec, clang_tidy_diff)) )
run_clang_tidy = os.path.join(
    config.test_source_root, "..", "clang-tidy", "tool", "run-clang-tidy.py")
config.substitutions.append(
    ('%run_clang_tidy',
     '%s %s' % (python_exec, run_clang_tidy)) )

config.substitutions.append(('%llvmshlibdir', config.clang_libs_dir))
config.substitutions.append(('%pluginext', config.llvm_plugin_ext))

# Plugins (loadable modules)
if config.has_plugins and config.llvm_plugin_ext:
    config.available_features.add('plugins')
