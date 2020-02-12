# -*- Python -*-

import os

def get_required_attr(config, attr_name):
  attr_value = getattr(config, attr_name, None)
  if not attr_value:
    lit_config.fatal(
      "No attribute %r in test configuration! You may need to run "
      "tests from your build directory or add this attribute "
      "to lit.site.cfg.py " % attr_name)
  return attr_value

# Setup config name.
config.name = 'ThreadSanitizer' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Setup environment variables for running ThreadSanitizer.
default_tsan_opts = "atexit_sleep_ms=0"

if config.host_os == 'Darwin':
  # On Darwin, we default to `abort_on_error=1`, which would make tests run
  # much slower. Let's override this and run lit tests with 'abort_on_error=0'.
  default_tsan_opts += ':abort_on_error=0'
  # On Darwin, we default to ignore_noninstrumented_modules=1, which also
  # suppresses some races the tests are supposed to find. Let's run without this
  # setting, but turn it back on for Darwin tests (see Darwin/lit.local.cfg.py).
  default_tsan_opts += ':ignore_noninstrumented_modules=0'

# Platform-specific default TSAN_OPTIONS for lit tests.
if default_tsan_opts:
  config.environment['TSAN_OPTIONS'] = default_tsan_opts
  default_tsan_opts += ':'
config.substitutions.append(('%env_tsan_opts=',
                             'env TSAN_OPTIONS=' + default_tsan_opts))

# GCC driver doesn't add necessary compile/link flags with -fsanitize=thread.
if config.compiler_id == 'GNU':
  extra_cflags = ["-fPIE", "-pthread", "-ldl", "-lrt", "-pie"]
else:
  extra_cflags = []

tsan_incdir = config.test_source_root + "/../"
# Setup default compiler flags used with -fsanitize=thread option.
clang_tsan_cflags = (["-fsanitize=thread",
                      "-Wall"] +
                      [config.target_cflags] +
                      config.debug_info_flags +
                      extra_cflags +
                      ["-I%s" % tsan_incdir])
clang_tsan_cxxflags = config.cxx_mode_flags + clang_tsan_cflags + ["-std=c++11"] + ["-I%s" % tsan_incdir]
# Add additional flags if we're using instrumented libc++.
# Instrumented libcxx currently not supported on Darwin.
if config.has_libcxx and config.host_os != 'Darwin':
  # FIXME: Dehardcode this path somehow.
  libcxx_path = os.path.join(config.compiler_rt_obj_root, "lib",
                             "tsan", "libcxx_tsan_%s" % config.target_arch)
  libcxx_incdir = os.path.join(libcxx_path, "include", "c++", "v1")
  libcxx_libdir = os.path.join(libcxx_path, "lib")
  libcxx_a = os.path.join(libcxx_libdir, "libc++.a")
  clang_tsan_cxxflags += ["-nostdinc++",
                          "-I%s" % libcxx_incdir]
  config.substitutions.append( ("%link_libcxx_tsan", libcxx_a) )
else:
  config.substitutions.append( ("%link_libcxx_tsan", "") )

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

config.substitutions.append( ("%clang_tsan ", build_invocation(clang_tsan_cflags)) )
config.substitutions.append( ("%clangxx_tsan ", build_invocation(clang_tsan_cxxflags)) )

# Define CHECK-%os to check for OS-dependent output.
config.substitutions.append( ('CHECK-%os', ("CHECK-" + config.host_os)))

config.substitutions.append( ("%deflake ", os.path.join(os.path.dirname(__file__), "deflake.bash") + " " + config.deflake_threshold))

# Default test suffixes.
config.suffixes = ['.c', '.cpp', '.m', '.mm']

if config.host_os not in ['FreeBSD', 'Linux', 'Darwin', 'NetBSD']:
  config.unsupported = True

if config.android:
  config.unsupported = True

if not config.parallelism_group:
  config.parallelism_group = 'shadow-memory'

if config.host_os == 'NetBSD':
  config.substitutions.insert(0, ('%run', config.netbsd_noaslr_prefix))
