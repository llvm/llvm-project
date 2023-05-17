# -*- Python -*-

import os
import re

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if attr_value == None:
    lit_config.fatal(
      "No attribute %r in test configuration! You may need to run "
      "tests from your build directory or add this attribute "
      "to lit.site.cfg.py " % attr_name)
  return attr_value

# Setup config name.
config.name = 'Profile-' + config.target_arch

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup executable root.
if hasattr(config, 'profile_lit_binary_dir') and \
        config.profile_lit_binary_dir is not None:
    config.test_exec_root = os.path.join(config.profile_lit_binary_dir, config.name)

target_is_msvc = bool(re.match(r'.*-windows-msvc$', config.target_triple))

if config.host_os in ['Linux']:
  extra_link_flags = ["-ldl"]
elif target_is_msvc:
  # InstrProf is incompatible with incremental linking. Disable it as a
  # workaround.
  extra_link_flags = ["-Wl,-incremental:no"]
else:
  extra_link_flags = []

# Test suffixes.
config.suffixes = ['.c', '.cpp', '.m', '.mm', '.ll', '.test']

# What to exclude.
config.excludes = ['Inputs']

# Clang flags.
target_cflags=[get_required_attr(config, "target_cflags")]
clang_cflags = target_cflags + extra_link_flags
clang_cxxflags = config.cxx_mode_flags + clang_cflags

# TODO: target_cflags can sometimes contain C++ only flags like -stdlib=<FOO>, which are
#       ignored when compiling as C code. Passing this flag when compiling as C results in
#       warnings that break tests that use -Werror.
#       We remove -stdlib= from the cflags here to avoid problems, but the interaction between
#       CMake and compiler-rt's tests should be reworked so that cflags don't contain C++ only
#       flags.
clang_cflags = [flag.replace('-stdlib=libc++', '').replace('-stdlib=libstdc++', '') for flag in clang_cflags]

def build_invocation(compile_flags, with_lto = False):
  lto_flags = []
  if with_lto and config.lto_supported:
    lto_flags += config.lto_flags
  return " " + " ".join([config.clang] + lto_flags + compile_flags) + " "

def exclude_unsupported_files_for_aix(dirname):
   for filename in os.listdir(dirname):
       source_path = os.path.join( dirname, filename)
       if os.path.isdir(source_path):
           continue
       f = open(source_path, 'r')
       try:
          data = f.read()
          # -fprofile-instr-generate and rpath are not supported on AIX, exclude all tests with them.
          if ("%clang_profgen" in data or "%clangxx_profgen" in data or "-rpath" in data):
            config.excludes += [ filename ]
       finally:
          f.close()

# Add clang substitutions.
config.substitutions.append( ("%clang ", build_invocation(clang_cflags)) )
config.substitutions.append( ("%clangxx ", build_invocation(clang_cxxflags)) )

config.substitutions.append( ("%clang_profgen ", build_invocation(clang_cflags) + " -fprofile-instr-generate ") )
config.substitutions.append( ("%clang_profgen=", build_invocation(clang_cflags) + " -fprofile-instr-generate=") )
config.substitutions.append( ("%clangxx_profgen ", build_invocation(clang_cxxflags) + " -fprofile-instr-generate ") )
config.substitutions.append( ("%clangxx_profgen=", build_invocation(clang_cxxflags) + " -fprofile-instr-generate=") )

config.substitutions.append( ("%clang_pgogen ", build_invocation(clang_cflags) + " -fprofile-generate ") )
config.substitutions.append( ("%clang_pgogen=", build_invocation(clang_cflags) + " -fprofile-generate=") )
config.substitutions.append( ("%clangxx_pgogen ", build_invocation(clang_cxxflags) + " -fprofile-generate ") )
config.substitutions.append( ("%clangxx_pgogen=", build_invocation(clang_cxxflags) + " -fprofile-generate=") )

config.substitutions.append( ("%clang_cspgogen ", build_invocation(clang_cflags) + " -fcs-profile-generate ") )
config.substitutions.append( ("%clang_cspgogen=", build_invocation(clang_cflags) + " -fcs-profile-generate=") )
config.substitutions.append( ("%clangxx_cspgogen ", build_invocation(clang_cxxflags) + " -fcs-profile-generate ") )
config.substitutions.append( ("%clangxx_cspgogen=", build_invocation(clang_cxxflags) + " -fcs-profile-generate=") )

config.substitutions.append( ("%clang_profuse=", build_invocation(clang_cflags) + " -fprofile-instr-use=") )
config.substitutions.append( ("%clangxx_profuse=", build_invocation(clang_cxxflags) + " -fprofile-instr-use=") )

config.substitutions.append( ("%clang_pgouse=", build_invocation(clang_cflags) + " -fprofile-use=") )
config.substitutions.append( ("%clangxx_profuse=", build_invocation(clang_cxxflags) + " -fprofile-instr-use=") )

config.substitutions.append( ("%clang_lto_profgen=", build_invocation(clang_cflags, True) + " -fprofile-instr-generate=") )

if config.host_os not in ['Windows', 'Darwin', 'FreeBSD', 'Linux', 'NetBSD', 'SunOS', 'AIX']:
  config.unsupported = True

if config.host_os in ['AIX']:
  config.available_features.add('system-aix')
  exclude_unsupported_files_for_aix(config.test_source_root)
  exclude_unsupported_files_for_aix(config.test_source_root + "/Posix")

if config.target_arch in ['armv7l']:
  config.unsupported = True

if config.android:
  config.unsupported = True
