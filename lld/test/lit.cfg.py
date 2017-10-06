# -*- Python -*-

import os
import platform
import re
import subprocess
import locale

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'lld'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.ll', '.s', '.test', '.yaml', '.objtxt']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.test_exec_root = os.path.join(config.lld_obj_root, 'test')

# Tweak the PATH to include the tools dir and the scripts dir.
llvm_config.with_environment('PATH',
                             [config.llvm_tools_dir, config.lld_tools_dir], append_path=True)

llvm_config.with_environment('LD_LIBRARY_PATH',
                             [config.lld_libs_dir, config.llvm_libs_dir], append_path=True)

llvm_config.use_default_substitutions()

# For each occurrence of a clang tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.lld_tools_dir, config.llvm_tools_dir]

tool_patterns = [
    ToolSubst('ld.lld', extra_args=['--full-shutdown']),
    'lld-link', 'llvm-as', 'llvm-mc', 'llvm-nm',
    'llvm-objdump', 'llvm-pdbutil', 'llvm-readobj', 'obj2yaml', 'yaml2obj',
    'lld']

llvm_config.add_tool_substitutions(tool_patterns, tool_dirs)

# When running under valgrind, we mangle '-vg' onto the end of the triple so we
# can check it with XFAIL and XTARGET.
if lit_config.useValgrind:
    config.target_triple += '-vg'

# Running on ELF based *nix
if platform.system() in ['FreeBSD', 'Linux']:
    config.available_features.add('system-linker-elf')

# Set if host-cxxabi's demangler can handle target's symbols.
if platform.system() not in ['Windows']:
    config.available_features.add('demangler')

llvm_config.feature_config(
    [('--build-mode', {'DEBUG': 'debug'}),
     ('--assertion-mode', {'ON': 'asserts'}),
     ('--targets-built', {'AArch64': 'aarch64',
                          'AMDGPU': 'amdgpu',
                          'ARM': 'arm',
                          'AVR': 'avr',
                          'Mips': 'mips',
                          'PowerPC': 'ppc',
                          'Sparc': 'sparc',
                          'X86': 'x86'})
     ])

# Set a fake constant version so that we get consitent output.
config.environment['LLD_VERSION'] = 'LLD 1.0'

# Indirectly check if the mt.exe Microsoft utility exists by searching for
# cvtres, which always accompanies it.  Alternatively, check if we can use
# libxml2 to merge manifests.
if (lit.util.which('cvtres', config.environment['PATH'])) or \
        (config.llvm_libxml2_enabled == '1'):
    config.available_features.add('manifest_tool')

if (config.llvm_libxml2_enabled == '1'):
    config.available_features.add('libxml2')
