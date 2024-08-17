// RUN: rm -rf %t && mkdir -p %t

// Check if -fcasid-output works on a cache miss with file based caching
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_DISABLE_MCCAS=1 %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fcasid-output -g -c %s -o %t/test.o
// RUN: cat %t/test.o | FileCheck %s
// RUN: rm -rf %t/test.o
// Check if -fcasid-output works on a cache hit with file based caching
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_DISABLE_MCCAS=1 %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fcasid-output -g -c %s -o %t/test.o
// RUN: cat %t/test.o | FileCheck %s
// RUN: rm -rf %t/test.o
// RUN: rm -rf %t/cas

// Check if -fcasid-output works on a cache miss with MCCAS
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fcasid-output -g -c %s -o %t/test.o
// RUN: cat %t/test.o | FileCheck %s
// RUN: rm -rf %t/test.o

// Check if -fcasid-output works on a cache hit with MCCAS
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang -target x86_64-apple-macos11 -Xclang -fcasid-output -g -c %s -o %t/test.o
// RUN: cat %t/test.o | FileCheck %s

// CHECK: llvmcas://{{[a-f0-9]+}}


void foo() {}
