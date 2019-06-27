# -*- Python -*-

import os

# Setup config name.
config.name = 'DataFlowSanitizer' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup default compiler flags used with -fsanitize=dataflow option.
clang_dfsan_cflags = ["-fsanitize=dataflow", config.target_cflags]
clang_dfsan_cxxflags = config.cxx_mode_flags + clang_dfsan_cflags

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

config.substitutions.append( ("%clang_dfsan ", build_invocation(clang_dfsan_cflags)) )
config.substitutions.append( ("%clangxx_dfsan ", build_invocation(clang_dfsan_cxxflags)) )

# Default test suffixes.
config.suffixes = ['.c', '.cc', '.cpp']

# DataFlowSanitizer tests are currently supported on Linux only.
if config.host_os not in ['Linux']:
  config.unsupported = True
