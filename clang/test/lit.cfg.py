# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Clang'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.i', '.cppm', '.m', '.mm', '.cu', '.hip',
                   '.ll', '.cl', '.clcpp', '.s', '.S', '.modulemap', '.test', '.rs', '.ifs', '.rc']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'debuginfo-tests']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.clang_obj_root, 'test')

llvm_config.use_default_substitutions()

llvm_config.use_clang()

config.substitutions.append(
    ('%src_include_dir', config.clang_src_dir + '/include'))

config.substitutions.append(
    ('%target_triple', config.target_triple))

# Propagate path to symbolizer for ASan/MSan.
llvm_config.with_system_environment(
    ['ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH'])

config.substitutions.append(('%PATH%', config.environment['PATH']))


# For each occurrence of a clang tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.clang_tools_dir, config.llvm_tools_dir]

tools = [
    'apinotes-test', 'c-index-test', 'clang-diff', 'clang-format', 'clang-repl',
    'clang-tblgen', 'clang-scan-deps', 'opt', 'llvm-ifs', 'yaml2obj',
    ToolSubst('%clang_extdef_map', command=FindTool(
        'clang-extdef-mapping'), unresolved='ignore'),
]

if config.clang_examples:
    config.available_features.add('examples')

def have_host_jit_support():
    clang_repl_exe = lit.util.which('clang-repl', config.clang_tools_dir)

    if not clang_repl_exe:
        print('clang-repl not found')
        return False

    try:
        clang_repl_cmd = subprocess.Popen(
            [clang_repl_exe, '--host-supports-jit'], stdout=subprocess.PIPE)
    except OSError:
        print('could not exec clang-repl')
        return False

    clang_repl_out = clang_repl_cmd.stdout.read().decode('ascii')
    clang_repl_cmd.wait()

    return 'true' in clang_repl_out

if have_host_jit_support():
    config.available_features.add('host-supports-jit')

if config.clang_staticanalyzer:
    config.available_features.add('staticanalyzer')
    tools.append('clang-check')

    if config.clang_staticanalyzer_z3 == '1':
        config.available_features.add('z3')

    check_analyzer_fixit_path = os.path.join(
        config.test_source_root, "Analysis", "check-analyzer-fixit.py")
    config.substitutions.append(
        ('%check_analyzer_fixit',
         '"%s" %s' % (config.python_executable, check_analyzer_fixit_path)))

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.substitutions.append(
    ('%hmaptool', "'%s' %s" % (config.python_executable,
                             os.path.join(config.clang_tools_dir, 'hmaptool'))))

config.substitutions.append(('%host_cc', config.host_cc))
config.substitutions.append(('%host_cxx', config.host_cxx))


# Plugins (loadable modules)
if config.has_plugins and config.llvm_plugin_ext:
    config.available_features.add('plugins')

# Set available features we allow tests to conditionalize on.
#
if config.clang_default_cxx_stdlib != '':
    config.available_features.add('default-cxx-stdlib-set')

# As of 2011.08, crash-recovery tests still do not pass on FreeBSD.
if platform.system() not in ['FreeBSD']:
    config.available_features.add('crash-recovery')

# Support for new pass manager.
if config.enable_experimental_new_pass_manager:
    config.available_features.add('experimental-new-pass-manager')

# ANSI escape sequences in non-dumb terminal
if platform.system() not in ['Windows']:
    config.available_features.add('ansi-escape-sequences')

# Capability to print utf8 to the terminal.
# Windows expects codepage, unless Wide API.
if platform.system() not in ['Windows']:
    config.available_features.add('utf8-capable-terminal')

# Support for libgcc runtime. Used to rule out tests that require
# clang to run with -rtlib=libgcc.
if platform.system() not in ['Darwin', 'Fuchsia']:
    config.available_features.add('libgcc')

# Case-insensitive file system


def is_filesystem_case_insensitive():
    handle, path = tempfile.mkstemp(
        prefix='case-test', dir=config.test_exec_root)
    isInsensitive = os.path.exists(
        os.path.join(
            os.path.dirname(path),
            os.path.basename(path).upper()
        ))
    os.close(handle)
    os.remove(path)
    return isInsensitive


if is_filesystem_case_insensitive():
    config.available_features.add('case-insensitive-filesystem')

# Tests that require the /dev/fd filesystem.
if os.path.exists('/dev/fd/0') and sys.platform not in ['cygwin']:
    config.available_features.add('dev-fd-fs')

# Set on native MS environment.
if re.match(r'.*-(windows-msvc)$', config.target_triple):
    config.available_features.add('ms-sdk')

# [PR8833] LLP64-incompatible tests
if not re.match(r'^x86_64.*-(windows-msvc|windows-gnu)$', config.target_triple):
    config.available_features.add('LP64')

# [PR12920] "clang-driver" -- set if gcc driver is not used.
if not re.match(r'.*-(cygwin)$', config.target_triple):
    config.available_features.add('clang-driver')

# Tests that are specific to the Apple Silicon macOS.
if re.match(r'^arm64(e)?-apple-(macos|darwin)', config.target_triple):
    config.available_features.add('apple-silicon-mac')

# [PR18856] Depends to remove opened file. On win32, a file could be removed
# only if all handles were closed.
if platform.system() not in ['Windows']:
    config.available_features.add('can-remove-opened-file')

# *-apple-macosx should also be XFAILED when 'darwin' is XFAILED.
if lit.util.isMacOSTriple(config.target_triple):
   config.available_features.add('darwin')

# Features
known_arches = ["x86_64", "mips64", "ppc64", "aarch64"]
if (any(config.target_triple.startswith(x) for x in known_arches)):
  config.available_features.add("clang-target-64-bits")


def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        features.append(arch.lower() + '-registered-target')
    return features


llvm_config.feature_config(
    [('--assertion-mode', {'ON': 'asserts'}),
     ('--cxxflags', {r'-D_GLIBCXX_DEBUG\b': 'libstdcxx-safe-mode'}),
     ('--targets-built', calculate_arch_features),
     ])

if lit.util.which('xmllint'):
    config.available_features.add('xmllint')

if config.enable_backtrace:
    config.available_features.add('backtrace')

if config.enable_threads:
    config.available_features.add('thread_support')

# Check if we should allow outputs to console.
run_console_tests = int(lit_config.params.get('enable_console', '0'))
if run_console_tests != 0:
    config.available_features.add('console')

lit.util.usePlatformSdkOnDarwin(config, lit_config)
macOSSDKVersion = lit.util.findPlatformSdkVersionOnMacOS(config, lit_config)
if macOSSDKVersion is not None:
    config.available_features.add('macos-sdk-' + str(macOSSDKVersion))

if os.path.exists('/etc/gentoo-release'):
    config.available_features.add('gentoo')

if config.enable_shared:
    config.available_features.add("enable_shared")

# Add a vendor-specific feature.
if config.clang_vendor_uti:
    config.available_features.add('clang-vendor=' + config.clang_vendor_uti)
