// RUN: %clang -cc1 -triple arm64-apple-ios6.0.0 -isysroot %S/doesnotexist %s 2>&1 | FileCheck %s
// RUN: %clang -cc1 -triple arm64-apple-ios6.0.0 -isysroot %S/doesnotexist -stdlib=libc++ %s -verify
// RUN: %clang -cc1 -x c++-cpp-output -triple arm64-apple-ios6.0.0 -isysroot %S/doesnotexist %s -verify
// CHECK: include path for stdlibc++ headers not found; pass '-stdlib=libc++' on the command line to use the libc++ standard library instead

// expected-no-diagnostics
