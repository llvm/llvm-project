# -*- Python -*-

import os

# Setup config name.
config.name = 'ORC' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

def build_invocation(compile_flags):
  return ' ' + ' '.join([config.clang] + compile_flags) + ' '

# Assume that llvm-jitlink is in the config.llvm_tools_dir.
llvm_jitlink = os.path.join(config.llvm_tools_dir, 'llvm-jitlink')
if config.host_os == 'Darwin':
  orc_rt_path = '%s/libclang_rt.orc_osx.a' % config.compiler_rt_libdir
else:
  orc_rt_path = '%s/libclang_rt.orc%s.a' % (config.compiler_rt_libdir, config.target_suffix)

if config.libunwind_shared:
  config.available_features.add('libunwind-available')
  shared_libunwind_path = os.path.join(config.libunwind_install_dir, 'libunwind.so')
  config.substitutions.append( ("%shared_libunwind", shared_libunwind_path) )

config.substitutions.append(
    ('%clang ', build_invocation([config.target_cflags])))
config.substitutions.append(
    ('%clangxx ',
     build_invocation(config.cxx_mode_flags + [config.target_cflags])))
config.substitutions.append(
    ('%llvm_jitlink', (llvm_jitlink + ' -orc-runtime=' + orc_rt_path)))

# Default test suffixes.
config.suffixes = ['.c', '.cpp', '.S']

# Exclude Inputs directories.
config.excludes = ['Inputs']

if config.host_os not in ['Darwin', 'FreeBSD', 'Linux']:
  config.unsupported = True
