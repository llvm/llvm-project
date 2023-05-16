# -*- Python -*-

import os
import platform
import re

import lit.formats

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if attr_value == None:
    lit_config.fatal(
      "No attribute %r in test configuration! You may need to run "
      "tests from your build directory or add this attribute "
      "to lit.site.cfg.py " % attr_name)
  return attr_value

# Setup config name.
config.name = 'MemProfiler' + config.name_suffix

# Platform-specific default MEMPROF_OPTIONS for lit tests.
default_memprof_opts = list(config.default_sanitizer_opts)

default_memprof_opts_str = ':'.join(default_memprof_opts)
if default_memprof_opts_str:
  config.environment['MEMPROF_OPTIONS'] = default_memprof_opts_str
  default_memprof_opts_str += ':'
config.substitutions.append(('%env_memprof_opts=',
                             'env MEMPROF_OPTIONS=' + default_memprof_opts_str))

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

libdl_flag = '-ldl'

# Setup default compiler flags used with -fmemory-profile option.
# FIXME: Review the set of required flags and check if it can be reduced.
target_cflags = [get_required_attr(config, 'target_cflags')]
target_cxxflags = config.cxx_mode_flags + target_cflags
clang_memprof_static_cflags = (['-fmemory-profile',
                            '-mno-omit-leaf-frame-pointer',
                            '-fno-omit-frame-pointer',
                            '-fno-optimize-sibling-calls'] +
                            config.debug_info_flags + target_cflags)
clang_memprof_static_cxxflags = config.cxx_mode_flags + clang_memprof_static_cflags

memprof_dynamic_flags = []
if config.memprof_dynamic:
  memprof_dynamic_flags = ['-shared-libsan']
  config.available_features.add('memprof-dynamic-runtime')
else:
  config.available_features.add('memprof-static-runtime')
clang_memprof_cflags = clang_memprof_static_cflags + memprof_dynamic_flags
clang_memprof_cxxflags = clang_memprof_static_cxxflags + memprof_dynamic_flags

def build_invocation(compile_flags):
  return ' ' + ' '.join([config.clang] + compile_flags) + ' '

config.substitutions.append( ("%clang ", build_invocation(target_cflags)) )
config.substitutions.append( ("%clangxx ", build_invocation(target_cxxflags)) )
config.substitutions.append( ("%clang_memprof ", build_invocation(clang_memprof_cflags)) )
config.substitutions.append( ("%clangxx_memprof ", build_invocation(clang_memprof_cxxflags)) )
if config.memprof_dynamic:
  shared_libmemprof_path = os.path.join(config.compiler_rt_libdir, 'libclang_rt.memprof{}.so'.format(config.target_suffix))
  config.substitutions.append( ("%shared_libmemprof", shared_libmemprof_path) )
  config.substitutions.append( ("%clang_memprof_static ", build_invocation(clang_memprof_static_cflags)) )
  config.substitutions.append( ("%clangxx_memprof_static ", build_invocation(clang_memprof_static_cxxflags)) )

config.substitutions.append( ("%libdl", libdl_flag) )

config.available_features.add('memprof-' + config.bits + '-bits')

config.available_features.add('fast-unwinder-works')

# Set LD_LIBRARY_PATH to pick dynamic runtime up properly.
new_ld_library_path = os.path.pathsep.join(
  (config.compiler_rt_libdir, config.environment.get('LD_LIBRARY_PATH', '')))
config.environment['LD_LIBRARY_PATH'] = new_ld_library_path

# Default test suffixes.
config.suffixes = ['.c', '.cpp']

config.substitutions.append(('%fPIC', '-fPIC'))
config.substitutions.append(('%fPIE', '-fPIE'))
config.substitutions.append(('%pie', '-pie'))

# Only run the tests on supported OSs.
if config.host_os not in ['Linux']:
  config.unsupported = True

if not config.parallelism_group:
  config.parallelism_group = 'shadow-memory'
