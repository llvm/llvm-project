# -*- Python -*-

import os

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

if config.host_os in ['Linux']:
  extra_link_flags = ["-ldl"]
elif config.host_os in ['Windows']:
  # InstrProf is incompatible with incremental linking. Disable it as a
  # workaround.
  extra_link_flags = ["-Wl,-incremental:no"]
else:
  extra_link_flags = []

# Test suffixes.
config.suffixes = ['.c', '.cc', '.cpp', '.m', '.mm', '.ll', '.test']

# What to exclude.
config.excludes = ['Inputs']

# Clang flags.
target_cflags=[get_required_attr(config, "target_cflags")]
clang_cflags = target_cflags + extra_link_flags
clang_cxxflags = config.cxx_mode_flags + clang_cflags

def build_invocation(compile_flags, with_lto = False):
  lto_flags = []
  lto_prefix = []
  if with_lto and config.lto_supported:
    lto_flags += config.lto_flags
    lto_prefix += config.lto_launch
  return " " + " ".join(lto_prefix + [config.clang] + lto_flags + compile_flags) + " "

# Add clang substitutions.
config.substitutions.append( ("%clang ", build_invocation(clang_cflags)) )
config.substitutions.append( ("%clangxx ", build_invocation(clang_cxxflags)) )
config.substitutions.append( ("%clang_profgen ", build_invocation(clang_cflags) + " -fprofile-instr-generate ") )
config.substitutions.append( ("%clang_profgen=", build_invocation(clang_cflags) + " -fprofile-instr-generate=") )
config.substitutions.append( ("%clang_pgogen ", build_invocation(clang_cflags) + " -fprofile-generate ") )
config.substitutions.append( ("%clang_pgogen=", build_invocation(clang_cflags) + " -fprofile-generate=") )

config.substitutions.append( ("%clangxx_profgen ", build_invocation(clang_cxxflags) + " -fprofile-instr-generate ") )
config.substitutions.append( ("%clangxx_profgen=", build_invocation(clang_cxxflags) + " -fprofile-instr-generate=") )
config.substitutions.append( ("%clangxx_pgogen ", build_invocation(clang_cxxflags) + " -fprofile-generate ") )
config.substitutions.append( ("%clangxx_pgogen=", build_invocation(clang_cxxflags) + " -fprofile-generate=") )

config.substitutions.append( ("%clang_profgen_gcc=", build_invocation(clang_cflags) + " -fprofile-generate=") )
config.substitutions.append( ("%clang_profuse_gcc=", build_invocation(clang_cflags) + " -fprofile-use=") )

config.substitutions.append( ("%clang_profuse=", build_invocation(clang_cflags) + " -fprofile-instr-use=") )
config.substitutions.append( ("%clangxx_profuse=", build_invocation(clang_cxxflags) + " -fprofile-instr-use=") )

config.substitutions.append( ("%clang_lto_profgen=", build_invocation(clang_cflags, True) + " -fprofile-instr-generate=") )

if config.host_os not in ['Windows', 'Darwin', 'FreeBSD', 'Linux', 'NetBSD', 'SunOS']:
  config.unsupported = True

if config.target_arch in ['armv7l']:
  config.unsupported = True

if config.android:
  config.unsupported = True
