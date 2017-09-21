# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Clang'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.c', '.cpp', '.cppm', '.m', '.mm', '.cu', '.ll', '.cl', '.s', '.S', '.modulemap', '.test', '.rs']

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.clang_obj_root, 'test')

# Clear some environment variables that might affect Clang.
#
# This first set of vars are read by Clang, but shouldn't affect tests
# that aren't specifically looking for these features, or are required
# simply to run the tests at all.
#
# FIXME: Should we have a tool that enforces this?

# safe_env_vars = ('TMPDIR', 'TEMP', 'TMP', 'USERPROFILE', 'PWD',
#                  'MACOSX_DEPLOYMENT_TARGET', 'IPHONEOS_DEPLOYMENT_TARGET',
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

llvm_config.clear_environment(possibly_dangerous_env_vars)

# Tweak the PATH to include the tools dir and the scripts dir.
llvm_config.with_environment('PATH', [config.llvm_tools_dir, config.clang_tools_dir], append_path=True)

llvm_config.with_environment('LD_LIBRARY_PATH', [config.llvm_shlib_dir, config.llvm_libs_dir], append_path=True)

# Propagate path to symbolizer for ASan/MSan.
llvm_config.with_system_environment(['ASAN_SYMBOLIZER_PATH', 'MSAN_SYMBOLIZER_PATH'])

# Discover the 'clang' and 'clangcc' to use.

def inferClang(PATH):
    # Determine which clang to use.
    clang = os.getenv('CLANG')

    # If the user set clang in the environment, definitely use that and don't
    # try to validate.
    if clang:
        return clang

    # Otherwise look in the path.
    clang = lit.util.which('clang', PATH)

    if not clang:
        lit_config.fatal("couldn't find 'clang' program, try setting "
                         "CLANG in your environment")

    return clang

config.clang = inferClang(config.environment['PATH']).replace('\\', '/')
if not lit_config.quiet:
    lit_config.note('using clang: %r' % config.clang)

# Plugins (loadable modules)
# TODO: This should be supplied by Makefile or autoconf.
if sys.platform in ['win32', 'cygwin']:
    has_plugins = config.enable_shared
else:
    has_plugins = True

if has_plugins and config.llvm_plugin_ext:
    config.available_features.add('plugins')

config.substitutions.append( ('%llvmshlibdir', config.llvm_shlib_dir) )
config.substitutions.append( ('%pluginext', config.llvm_plugin_ext) )
config.substitutions.append( ('%PATH%', config.environment['PATH']) )

if config.clang_examples:
    config.available_features.add('examples')

# Note that when substituting %clang_cc1 also fill in the include directory of
# the builtin headers. Those are part of even a freestanding environment, but
# Clang relies on the driver to locate them.
def getClangBuiltinIncludeDir(clang):
    # FIXME: Rather than just getting the version, we should have clang print
    # out its resource dir here in an easy to scrape form.
    cmd = subprocess.Popen([clang, '-print-file-name=include'],
                           stdout=subprocess.PIPE,
                           env=config.environment)
    if not cmd.stdout:
      lit_config.fatal("Couldn't find the include dir for Clang ('%s')" % clang)
    dir = cmd.stdout.read().strip()
    if sys.platform in ['win32'] and not llvm_config.use_lit_shell:
        # Don't pass dosish path separator to msys bash.exe.
        dir = dir.replace('\\', '/')
    # Ensure the result is an ascii string, across Python2.5+ - Python3.
    return str(dir.decode('ascii'))

def makeItaniumABITriple(triple):
    m = re.match(r'(\w+)-(\w+)-(\w+)', triple)
    if not m:
      lit_config.fatal("Could not turn '%s' into Itanium ABI triple" % triple)
    if m.group(3).lower() != 'win32':
      # All non-win32 triples use the Itanium ABI.
      return triple
    return m.group(1) + '-' + m.group(2) + '-mingw32'

def makeMSABITriple(triple):
    m = re.match(r'(\w+)-(\w+)-(\w+)', triple)
    if not m:
      lit_config.fatal("Could not turn '%s' into MS ABI triple" % triple)
    isa = m.group(1).lower()
    vendor = m.group(2).lower()
    os = m.group(3).lower()
    if os == 'win32':
      # If the OS is win32, we're done.
      return triple
    if isa.startswith('x86') or isa == 'amd64' or re.match(r'i\d86', isa):
      # For x86 ISAs, adjust the OS.
      return isa + '-' + vendor + '-win32'
    # -win32 is not supported for non-x86 targets; use a default.
    return 'i686-pc-win32'

config.substitutions.append( ('%clang_analyze_cc1',
                              '%clang_cc1 -analyze %analyze') )
config.substitutions.append( ('%clang_cc1',
                              '%s -cc1 -internal-isystem %s -nostdsysteminc'
                              % (config.clang,
                                 getClangBuiltinIncludeDir(config.clang))) )
config.substitutions.append( ('%clang_cpp', ' ' + config.clang +
                              ' --driver-mode=cpp '))
config.substitutions.append( ('%clang_cl', ' ' + config.clang +
                              ' --driver-mode=cl '))
config.substitutions.append( ('%clangxx', ' ' + config.clang +
                              ' --driver-mode=g++ '))
config.substitutions.append( ('%clang', ' ' + config.clang + ' ') )
config.substitutions.append( ('%test_debuginfo', ' ' + config.llvm_src_root + '/utils/test_debuginfo.pl ') )
config.substitutions.append( ('%itanium_abi_triple', makeItaniumABITriple(config.target_triple)) )
config.substitutions.append( ('%ms_abi_triple', makeMSABITriple(config.target_triple)) )
config.substitutions.append( ('%resource_dir', getClangBuiltinIncludeDir(config.clang)) )
config.substitutions.append( ('%python', config.python_executable) )

# The host triple might not be set, at least if we're compiling clang from
# an already installed llvm.
if config.host_triple and config.host_triple != '@LLVM_HOST_TRIPLE@':
    config.substitutions.append( ('%target_itanium_abi_host_triple', '--target=%s' % makeItaniumABITriple(config.host_triple)) )
else:
    config.substitutions.append( ('%target_itanium_abi_host_triple', '') )

config.substitutions.append( ('%src_include_dir', config.clang_src_dir + '/include') )

# FIXME: Find nicer way to prohibit this.
config.substitutions.append(
    (' clang ', """*** Do not use 'clang' in tests, use '%clang'. ***""") )
config.substitutions.append(
    (' clang\+\+ ', """*** Do not use 'clang++' in tests, use '%clangxx'. ***"""))
config.substitutions.append(
    (' clang-cc ',
     """*** Do not use 'clang-cc' in tests, use '%clang_cc1'. ***""") )
config.substitutions.append(
    (' clang -cc1 -analyze ',
     """*** Do not use 'clang -cc1 -analyze' in tests, use '%clang_analyze_cc1'. ***""") )
config.substitutions.append(
    (' clang -cc1 ',
     """*** Do not use 'clang -cc1' in tests, use '%clang_cc1'. ***""") )
config.substitutions.append(
    (' %clang-cc1 ',
     """*** invalid substitution, use '%clang_cc1'. ***""") )
config.substitutions.append(
    (' %clang-cpp ',
     """*** invalid substitution, use '%clang_cpp'. ***""") )
config.substitutions.append(
    (' %clang-cl ',
     """*** invalid substitution, use '%clang_cl'. ***""") )

# For each occurrence of a clang tool name as its own word, replace it
# with the full path to the build directory holding that tool.  This
# ensures that we are testing the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = os.path.pathsep.join((config.clang_tools_dir, config.llvm_tools_dir))

# Regex assertions to reject neighbor hyphens/dots (seen in some tests).
# For example, don't match 'clang-check-' or '.clang-format'.
NoPreHyphenDot = r"(?<!(-|\.))"
NoPostHyphenDot = r"(?!(-|\.))"
NoPostBar = r"(?!(/|\\))"

tool_patterns = [r"\bFileCheck\b",
                 r"\bc-index-test\b",
                 NoPreHyphenDot + r"\bclang-check\b" + NoPostHyphenDot,
                 NoPreHyphenDot + r"\bclang-diff\b" + NoPostHyphenDot,
                 NoPreHyphenDot + r"\bclang-format\b" + NoPostHyphenDot,
                 # FIXME: Some clang test uses opt?
                 NoPreHyphenDot + r"\bopt\b" + NoPostBar + NoPostHyphenDot,
                 # Handle these specially as they are strings searched
                 # for during testing.
                 r"\| \bcount\b",
                 r"\| \bnot\b"]

if config.clang_examples:
    tool_patterns.append(NoPreHyphenDot + r"\bclang-interpreter\b" + NoPostHyphenDot)

for pattern in tool_patterns:
    # Extract the tool name from the pattern.  This relies on the tool
    # name being surrounded by \b word match operators.  If the
    # pattern starts with "| ", include it in the string to be
    # substituted.
    tool_match = re.match(r"^(\\)?((\| )?)\W+b([0-9A-Za-z-_]+)\\b\W*$",
                          pattern)
    tool_pipe = tool_match.group(2)
    tool_name = tool_match.group(4)
    tool_path = lit.util.which(tool_name, tool_dirs)
    if not tool_path:
        # Warn, but still provide a substitution.
        lit_config.note('Did not find ' + tool_name + ' in ' + tool_dirs)
        tool_path = config.clang_tools_dir + '/' + tool_name
    config.substitutions.append((pattern, tool_pipe + tool_path))

###

# Set available features we allow tests to conditionalize on.
#
if config.clang_default_cxx_stdlib != '':
    config.available_features.add('default-cxx-stdlib-set')

# Enabled/disabled features
if config.clang_staticanalyzer:
    config.available_features.add("staticanalyzer")

    if config.clang_staticanalyzer_z3 == '1':
        config.available_features.add("z3")

# As of 2011.08, crash-recovery tests still do not pass on FreeBSD.
if platform.system() not in ['FreeBSD']:
    config.available_features.add('crash-recovery')

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
    handle, path = tempfile.mkstemp(prefix='case-test', dir=config.test_exec_root)
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
if os.path.exists("/dev/fd/0") and sys.platform not in ['cygwin']:
    config.available_features.add('dev-fd-fs')

# Not set on native MS environment.
if not re.match(r'.*-win32$', config.target_triple):
    config.available_features.add('non-ms-sdk')

# Not set on native PS4 environment.
if not re.match(r'.*-scei-ps4', config.target_triple):
    config.available_features.add('non-ps4-sdk')

# [PR8833] LLP64-incompatible tests
if not re.match(r'^x86_64.*-(win32|mingw32|windows-gnu)$', config.target_triple):
    config.available_features.add('LP64')

# [PR12920] "clang-driver" -- set if gcc driver is not used.
if not re.match(r'.*-(cygwin)$', config.target_triple):
    config.available_features.add('clang-driver')

# [PR18856] Depends to remove opened file. On win32, a file could be removed
# only if all handles were closed.
if platform.system() not in ['Windows']:
    config.available_features.add('can-remove-opened-file')

def calculate_arch_features(arch_string):
    features = []
    for arch in arch_string.split():
        features.append(arch.lower() + '-registered-target')
    return features

llvm_config.feature_config(
  [('--assertion-mode', {'ON' : 'asserts'}),
   ('--cxxflags', {r'-D_GLIBCXX_DEBUG\b' : 'libstdcxx-safe-mode'}),
   ('--targets-built', calculate_arch_features)
  ])

if lit.util.which('xmllint'):
    config.available_features.add('xmllint')

if config.enable_backtrace:
    config.available_features.add("backtrace")

# Check if we should allow outputs to console.
run_console_tests = int(lit_config.params.get('enable_console', '0'))
if run_console_tests != 0:
  config.available_features.add('console')

lit.util.usePlatformSdkOnDarwin(config, lit_config)
macOSSDKVersion = lit.util.findPlatformSdkVersionOnMacOS(config, lit_config)
if macOSSDKVersion is not None:
    config.available_features.add('macos-sdk-' + macOSSDKVersion)
