#!/bin/sh

# UNSUPPORTED: !libclang-loadable

# Tests require libclang.so which is only built with LLVM_ENABLE_PIC=ON
#
# Covered by libclang-loadable, may need to augment test for lack of
# libclang.so.

# Do not try to run if libclang was built with sanitizers because
# the sanitizer library will likely be loaded too late to perform
# interception and will then fail.
# We could use LD_PRELOAD/DYLD_INSERT_LIBRARIES but this isn't
# portable so its easier just to not run the tests when building
# with ASan.
#
# FIXME: Handle !LLVM_USE_SANITIZER = "".
# lit.site.cfg.py has config.llvm_use_sanitizer = ""

# Tests fail on Windows, and need someone knowledgeable to fix.
# It's not clear whether it's a test or a valid binding problem.
# XFAIL: target={{.*windows.*}}

# The Python FFI interface is broken on AIX: https://bugs.python.org/issue38628.
# XFAIL: target={{.*-aix.*}}

# AArch64, Hexagon, and Sparc have known test failures that need to be
# addressed.
# SystemZ has broken Python/FFI interface:
# https://reviews.llvm.org/D52840#1265716
# XFAIL: target={{(aarch64|hexagon|sparc*|s390x)-.*}}

# Tests will fail if cross-compiling for a different target, as tests will try
# to use the host Python3_EXECUTABLE and make FFI calls to functions in target
# libraries.
#
# FIXME: Consider a solution that allows better control over these tests in
# a crosscompiling scenario. e.g. registering them with lit to allow them to
# be explicitly skipped via appropriate LIT_ARGS, or adding a mechanism to
# allow specifying a python interpreter compiled for the target that could
# be executed using qemu-user.
#
# FIXME: Handle CMAKE_CROSSCOMPILING.
# Again, might already be handled by libclang-loadable.

# RUN: env PYTHONPATH=%S/../../../bindings/python \
# RUN:   CLANG_LIBRARY_PATH=`llvm-config --libdir` \
# RUN:   %python -m unittest discover -s %S/tests
