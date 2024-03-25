# -*- Python -*-

import os
import platform
import re

import lit.formats

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if attr_value is None:
    lit_config.fatal(
      'No attribute %r in test configuration! You may need to run '
      'tests from your build directory or add this attribute '
      'to lit.site.cfg.py ' % attr_name)
  return attr_value

# Setup config name.
config.name = 'AddressSanitizerABI' + config.name_suffix

# Platform-specific default ASAN_ABI_OPTIONS for lit tests.
default_asan_abi_opts = list(config.default_sanitizer_opts)

default_asan_abi_opts_str = ':'.join(default_asan_abi_opts)
if default_asan_abi_opts_str:
  config.environment['ASAN_ABI_OPTIONS'] = default_asan_abi_opts_str
  default_asan_abi_opts_str += ':'
config.substitutions.append(('%env_asan_abi_opts=',
                             'env ASAN_ABI_OPTIONS=' + default_asan_abi_opts_str))

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# GCC-ASan doesn't link in all the necessary libraries automatically, so
# we have to do it ourselves.
extra_link_flags = []

# Setup default compiler flags used with -fsanitize=address option.
# FIXME: Review the set of required flags and check if it can be reduced.
target_cflags = [get_required_attr(config, 'target_cflags')] + extra_link_flags
target_cxxflags = config.cxx_mode_flags + target_cflags
clang_asan_abi_static_cflags = (['-fsanitize=address',
                            '-fsanitize-stable-abi',
                            '-mno-omit-leaf-frame-pointer',
                            '-fno-omit-frame-pointer',
                            '-fno-optimize-sibling-calls'] +
                            config.debug_info_flags + target_cflags)
clang_asan_abi_static_cxxflags = config.cxx_mode_flags + clang_asan_abi_static_cflags

config.available_features.add('asan_abi-static-runtime')
clang_asan_abi_cflags = clang_asan_abi_static_cflags
clang_asan_abi_cxxflags = clang_asan_abi_static_cxxflags

def build_invocation(compile_flags):
  return ' ' + ' '.join([config.clang] + compile_flags) + ' '

config.substitutions.append( ('%clang ', build_invocation(target_cflags)) )
config.substitutions.append( ('%clangxx ', build_invocation(target_cxxflags)) )
config.substitutions.append( ('%clang_asan_abi ', build_invocation(clang_asan_abi_cflags)) )
config.substitutions.append( ('%clangxx_asan_abi ', build_invocation(clang_asan_abi_cxxflags)) )

libasan_abi_path = os.path.join(config.compiler_rt_libdir, 'libclang_rt.asan_abi_osx.a'.format(config.apple_platform))

if libasan_abi_path is not None:
  config.substitutions.append( ('%libasan_abi', libasan_abi_path) )
  config.substitutions.append( ('%clang_asan_abi_static ', build_invocation(clang_asan_abi_static_cflags)) )
  config.substitutions.append( ('%clangxx_asan_abi_static ', build_invocation(clang_asan_abi_static_cxxflags)) )

config.suffixes = ['.c', '.cpp']

if config.host_os == 'Darwin':
  config.suffixes.append('.mm')
else:
  config.unsupported = True
