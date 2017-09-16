# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import platform
import subprocess

import lit.formats
import lit.util

# name: The name of this test suite.
config.name = 'Clang-Unit'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = []

# test_source_root: The root path where tests are located.
# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.clang_obj_root, 'unittests')
config.test_source_root = config.test_exec_root

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, 'Tests')

# Propagate the temp directory. Windows requires this because it uses \Windows\
# if none of these are present.
if 'TMP' in os.environ:
    config.environment['TMP'] = os.environ['TMP']
if 'TEMP' in os.environ:
    config.environment['TEMP'] = os.environ['TEMP']

# Propagate path to symbolizer for ASan/MSan.
for symbolizer in ['ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH']:
    if symbolizer in os.environ:
        config.environment[symbolizer] = os.environ[symbolizer]

shlibpath_var = ''
if platform.system() == 'Linux':
    shlibpath_var = 'LD_LIBRARY_PATH'
elif platform.system() == 'Darwin':
    shlibpath_var = 'DYLD_LIBRARY_PATH'
elif platform.system() == 'Windows':
    shlibpath_var = 'PATH'

# in stand-alone builds, shlibdir is clang's build tree
# while llvm_libs_dir is installed LLVM (and possibly older clang)
shlibpath = os.path.pathsep.join((config.shlibdir, config.llvm_libs_dir,
                                 config.environment.get(shlibpath_var,'')))

config.environment[shlibpath_var] = shlibpath
