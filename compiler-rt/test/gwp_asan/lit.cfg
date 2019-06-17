# -*- Python -*-

import os

# Setup config name.
config.name = 'GWP-ASan' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.c', '.cc', '.cpp', '.test']

# C & CXX flags.
c_flags = ([config.target_cflags])

# Android doesn't want -lrt.
if not config.android:
  c_flags += ["-lrt"]

cxx_flags = (c_flags + config.cxx_mode_flags + ["-std=c++11"])

gwp_asan_flags = ["-fsanitize=scudo"]

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

# Add substitutions.
config.substitutions.append(("%clang ", build_invocation(c_flags)))
config.substitutions.append(("%clang_gwp_asan ", build_invocation(c_flags + gwp_asan_flags)))
config.substitutions.append(("%clangxx_gwp_asan ", build_invocation(cxx_flags + gwp_asan_flags)))

# Platform-specific default GWP_ASAN for lit tests. Ensure that GWP-ASan is
# enabled and that it samples every allocation.
default_gwp_asan_options = 'Enabled=1:SampleRate=1'

config.environment['GWP_ASAN_OPTIONS'] = default_gwp_asan_options
default_gwp_asan_options += ':'
config.substitutions.append(('%env_gwp_asan_options=',
                             'env GWP_ASAN_OPTIONS=' + default_gwp_asan_options))

# GWP-ASan tests are currently supported on Linux only.
if config.host_os not in ['Linux']:
   config.unsupported = True
