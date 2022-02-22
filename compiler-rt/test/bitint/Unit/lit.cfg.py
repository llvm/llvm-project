# -*- Python -*-

import os

# Setup config name.
config.name = 'bitint' + config.name_suffix

config.test_source_root = os.path.dirname(__file__)

# Test suffixes.
config.suffixes = ['.c', '.cc', '.cpp', '.m', '.mm']

base_lib = os.path.join(config.compiler_rt_libdir, "libclang_rt.bitint%s.a"
                          % config.target_suffix)

config.substitutions.append( ("%libbitint ", base_lib + ' -lc ') )

def build_invocation(compile_flags):
  return " ".join([config.clang] + compile_flags) + " "

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if attr_value == None:
    lit_config.fatal(
      "No attribute %r in test configuration! You may need to run "
      "tests from your build directory or add this attribute "
      "to lit.site.cfg.py " % attr_name)
  return attr_value

builtins_source_dir = os.path.join(
  get_required_attr(config, "compiler_rt_src_root"), "lib", "builtins")
if sys.platform in ['win32'] and execute_external:
  # Don't pass dosish path separator to msys bash.exe.
  builtins_source_dir = builtins_source_dir.replace('\\', '/')

clang_bitint_static_cflags = (['-g', 
  '-fno-builtin',
  '-nodefaultlibs',
  '-I', builtins_source_dir,
  config.target_cflags])

config.substitutions.append( ("%clang_bitint ", \
                              build_invocation(clang_bitint_static_cflags)))
