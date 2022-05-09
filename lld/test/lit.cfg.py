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
# For now we require '&&' between commands, until they get globally killed and the test runner updated.
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

llvm_config.use_default_substitutions()
llvm_config.use_lld()

tool_patterns = [
    'llc', 'llvm-as', 'llvm-mc', 'llvm-nm', 'llvm-objdump', 'llvm-otool', 'llvm-pdbutil',
    'llvm-dwarfdump', 'llvm-readelf', 'llvm-readobj', 'obj2yaml', 'yaml2obj',
    'opt', 'llvm-dis']

llvm_config.add_tool_substitutions(tool_patterns)

# LLD tests tend to be flaky on NetBSD, so add some retries.
# We don't do this on other platforms because it's slower.
if platform.system() in ['NetBSD']:
    config.test_retry_attempts = 2

# When running under valgrind, we mangle '-vg' onto the end of the triple so we
# can check it with XFAIL and XTARGET.
if lit_config.useValgrind:
    config.target_triple += '-vg'

# Running on ELF based *nix
if platform.system() in ['FreeBSD', 'NetBSD', 'Linux']:
    config.available_features.add('system-linker-elf')

# Set if host-cxxabi's demangler can handle target's symbols.
if platform.system() not in ['Windows']:
    config.available_features.add('demangler')

llvm_config.feature_config(
    [('--targets-built', {'AArch64': 'aarch64',
                          'AMDGPU': 'amdgpu',
                          'ARM': 'arm',
                          'AVR': 'avr',
                          'Hexagon': 'hexagon',
                          'Mips': 'mips',
                          'MSP430': 'msp430',
                          'PowerPC': 'ppc',
                          'RISCV': 'riscv',
                          'Sparc': 'sparc',
                          'WebAssembly': 'wasm',
                          'X86': 'x86'}),
     ('--assertion-mode', {'ON': 'asserts'}),
     ])

# Set a fake constant version so that we get consistent output.
config.environment['LLD_VERSION'] = 'LLD 1.0'

# LLD_IN_TEST determines how many times `main` is run inside each process, which
# lets us test that it's cleaning up after itself and resetting global state
# correctly (which is important for usage as a library).
run_lld_main_twice = lit_config.params.get('RUN_LLD_MAIN_TWICE', False)
if not run_lld_main_twice:
    config.environment['LLD_IN_TEST'] = '1'
else:
    config.environment['LLD_IN_TEST'] = '2'
    # Many ELF tests fail in this mode.
    config.excludes.append('ELF')
    # Some old Mach-O backend tests fail, and it's due for removal anyway.
    config.excludes.append('mach-o')
    # Some new Mach-O backend tests fail; give them a way to mark themselves
    # unsupported in this mode.
    config.available_features.add('main-run-twice')

# Indirectly check if the mt.exe Microsoft utility exists by searching for
# cvtres, which always accompanies it.  Alternatively, check if we can use
# libxml2 to merge manifests.
if (lit.util.which('cvtres', config.environment['PATH']) or
        config.have_libxml2):
    config.available_features.add('manifest_tool')

if config.have_libxar:
    config.available_features.add('xar')

if config.have_libxml2:
    config.available_features.add('libxml2')

if config.have_dia_sdk:
    config.available_features.add("diasdk")

if config.sizeof_void_p == 8:
    config.available_features.add("llvm-64-bits")

if config.has_plugins:
    config.available_features.add('plugins')

if config.build_examples:
    config.available_features.add('examples')

if config.linked_bye_extension:
    config.substitutions.append(('%loadbye', ''))
    config.substitutions.append(('%loadnewpmbye', ''))
else:
    config.substitutions.append(('%loadbye',
                                 '-load={}/Bye{}'.format(config.llvm_shlib_dir,
                                                         config.llvm_shlib_ext)))
    config.substitutions.append(('%loadnewpmbye',
                                 '-load-pass-plugin={}/Bye{}'
                                 .format(config.llvm_shlib_dir,
                                         config.llvm_shlib_ext)))

tar_executable = lit.util.which('tar', config.environment['PATH'])
if tar_executable:
    env = os.environ
    env['LANG'] = 'C'
    tar_version = subprocess.Popen(
        [tar_executable, '--version'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env)
    sout, _ = tar_version.communicate()
    if 'GNU tar' in sout.decode():
        config.available_features.add('gnutar')

# ELF tests expect the default target for ld.lld to be ELF.
if config.ld_lld_default_mingw:
    config.excludes.append('ELF')
