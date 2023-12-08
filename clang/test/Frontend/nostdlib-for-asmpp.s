// RUN: %clang_cc1 -x assembler-with-cpp -triple arm64-apple-ios6.0.0 -isysroot %S/doesnotexist -std=c++11 -v %s 2>&1 | FileCheck %s
// The C++ stdlib path should not be included for an assembly source.

// CHECK-NOT: usr/include/c++/
// CHECK-NOT: include path for stdlibc++ headers not found; pass '-std=libc++' on the command line to use the libc++ standard library instead
