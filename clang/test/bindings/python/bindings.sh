#!/bin/sh

# UNSUPPORTED: !libclang-loadable

# Tests fail on Windows, and need someone knowledgeable to fix.
# It's not clear whether it's a test or a valid binding problem.
# XFAIL: target={{.*windows.*}}

# The Python FFI interface is broken on AIX: https://bugs.python.org/issue38628.
# XFAIL: target={{.*-aix.*}}

# Hexagon has known test failures that need to be addressed.
# https://reviews.llvm.org/D52840#1265716
# XFAIL: target={{hexagon-.*}}
# python SEGVs on Linux/sparc64 when loading libclang.so.  Seems to be an FFI
# issue, too.
# XFAIL: target={{sparc.*-.*-linux.*}}

# Tests will fail if cross-compiling for a different target, as tests will try
# to use the host Python3_EXECUTABLE and make FFI calls to functions in target
# libraries.
#
# FIXME: Consider a solution that allows better control over these tests in
# a crosscompiling scenario. e.g. registering them with lit to allow them to
# be explicitly skipped via appropriate LIT_ARGS, or adding a mechanism to
# allow specifying a python interpreter compiled for the target that could
# be executed using qemu-user.
# REQUIRES: native

# SystemZ has broken Python/FFI interface
# according to https://reviews.llvm.org/D52840#1265716
# This leads to failures only when Clang is built with GCC apparently, see:
# https://github.com/llvm/llvm-project/pull/146844#issuecomment-3048291798
# REQUIRES: !target={{s390x-.*}}

# RUN: env PYTHONPATH=%S/../../../bindings/python \
# RUN:   CLANG_LIBRARY_PATH=%libdir \
# RUN:   %python -m unittest discover -s %S/tests
