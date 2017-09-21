# -*- Python -*-

import os
import platform
import re
import subprocess
import locale

import lit.formats
import lit.util

from lit.llvm import llvm_config

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
llvm_config.with_environment('PATH', [config.llvm_tools_dir, config.lld_tools_dir], append_path=True)

llvm_config.with_environment('LD_LIBRARY_PATH', [config.lld_libs_dir, config.llvm_libs_dir], append_path=True)

# For each occurrence of a lld tool name as its own word, replace it
# with the full path to the build directory holding that tool.  This
# ensures that we are testing the tools just built and not some random
# tools that might happen to be in the user's PATH.

# Regex assertions to reject neighbor hyphens/dots (seen in some tests).
# For example, we want to prefix 'lld' and 'ld.lld' but not the 'lld' inside
# of 'ld.lld'.
NoPreJunk = r"(?<!(-|\.|/))"
NoPostJunk = r"(?!(-|\.))"

config.substitutions.append( (r"\bld.lld\b", 'ld.lld --full-shutdown') )

tool_patterns = [r"\bFileCheck\b",
                 r"\bnot\b",
                 NoPreJunk + r"\blld\b" + NoPostJunk,
                 r"\bld.lld\b",
                 r"\blld-link\b",
                 r"\bllvm-as\b",
                 r"\bllvm-mc\b",
                 r"\bllvm-nm\b",
                 r"\bllvm-objdump\b",
                 r"\bllvm-pdbutil\b",
                 r"\bllvm-readobj\b",
                 r"\bobj2yaml\b",
                 r"\byaml2obj\b"]

for pattern in tool_patterns:
    # Extract the tool name from the pattern.  This relies on the tool
    # name being surrounded by \b word match operators.  If the
    # pattern starts with "| ", include it in the string to be
    # substituted.
    tool_match = re.match(r"^(\\)?((\| )?)\W+b([0-9A-Za-z-_\.]+)\\b\W*$",
                          pattern)
    tool_pipe = tool_match.group(2)
    tool_name = tool_match.group(4)
    tool_path = lit.util.which(tool_name, config.environment['PATH'])
    if not tool_path:
        # Warn, but still provide a substitution.
        lit_config.note('Did not find ' + tool_name + ' in ' + path)
        tool_path = config.llvm_tools_dir + '/' + tool_name
    config.substitutions.append((pattern, tool_pipe + tool_path))

# Add site-specific substitutions.
config.substitutions.append( ('%python', config.python_executable) )

###

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
    [('--build-mode', {'DEBUG' : 'debug'}),
     ('--assertion-mode', {'ON' : 'asserts'}),
     ('--targets-built', {'AArch64' : 'aarch64',
                          'AMDGPU' : 'amdgpu',
                          'ARM' : 'arm',
                          'AVR' : 'avr',
                          'Mips' : 'mips',
                          'PowerPC' : 'ppc',
                          'Sparc' : 'sparc',
                          'X86' : 'x86'})
    ])

# Set a fake constant version so that we get consitent output.
config.environment['LLD_VERSION'] = 'LLD 1.0'

# Indirectly check if the mt.exe Microsoft utility exists by searching for
# cvtres, which always accompanies it.  Alternatively, check if we can use
# libxml2 to merge manifests.
if (lit.util.which('cvtres', config.environment['PATH'])) or \
 (config.llvm_libxml2_enabled == "1"):
    config.available_features.add('manifest_tool')

if (config.llvm_libxml2_enabled == "1"):
    config.available_features.add('libxml2')
