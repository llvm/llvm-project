// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -v -fsyntax-only -x c %s 2>&1 | FileCheck %s

// CHECK: -fdepscan
