// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t && mkdir -p %t

// RUN: not env LLVM_CACHE_CAS_PATH=%t/cas cache-build-session %clang-cache \
// RUN:   %clang -fsyntax-only -x c %s \
// RUN:   2>&1 | FileCheck %s

#include "missing.h"
// CHECK: fatal error: 'missing.h' file not found
