# -*- Python -*-

import os

# Setup config name.
config.name = 'ScudoStandalone' + config.name_suffix

# Setup source root.
config.test_source_root = os.path.dirname(__file__)

# Path to the shared library
shared_libscudo = os.path.join(config.compiler_rt_libdir,
    "libclang_rt.scudo_standalone%s.so" % config.target_suffix)

# Test suffixes.
config.suffixes = ['.c', '.cpp']
config.excludes = ['lit-unmigrated']

# C & CXX flags. For cross-compiling, make sure that we pick up the
# libclang_rt.scudo_standalone libraries from the working build directory (using
# `-resource-dir`), rather than the host compiler.
c_flags = ([config.target_cflags] +
           ["-pthread", "-fPIE", "-pie", "-O0", "-UNDEBUG", "-Wl,--gc-sections",
            "-resource-dir=" + config.compiler_rt_libdir + "/../../"])

cxx_flags = (c_flags + config.cxx_mode_flags + ["-std=c++14"])

scudo_flags = ["-fsanitize=scudo"]

def build_invocation(compile_flags):
  return " " + " ".join([config.clang] + compile_flags) + " "

# Add substitutions.
config.substitutions.append(("%clang ", build_invocation(c_flags)))
config.substitutions.append(("%clang_scudo ", build_invocation(c_flags + scudo_flags)))
config.substitutions.append(("%clangxx_scudo ", build_invocation(cxx_flags + scudo_flags)))
config.substitutions.append(("%shared_libscudo", shared_libscudo))

# Disable GWP-ASan for scudo internal tests.
default_scudo_opts = ''
if config.gwp_asan:
  default_scudo_opts += 'GWP_ASAN_Enabled=false:'

config.substitutions.append(('%env_scudo_opts=',
                             'env SCUDO_OPTIONS=' + default_scudo_opts))

# Hardened Allocator tests are currently supported on Linux only.
if config.host_os not in ['Linux']:
   config.unsupported = True
