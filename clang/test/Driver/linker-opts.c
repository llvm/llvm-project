// RUN: env LIBRARY_PATH=%T/test1 %clang -x c %s -### 2>&1 | FileCheck %s
// CHECK: "-L{{.*}}/test1"

// GCC driver is used as linker on cygming. It should be aware of LIBRARY_PATH.
// XFAIL: win32
// REQUIRES: clang-driver
// REQUIRES: native

// Make sure that LIBRARY_PATH works for both i386 and x86_64 on Darwin.
// RUN: env LIBRARY_PATH=%T/test1 %clang -target x86_64-apple-darwin %s -### 2>&1 | FileCheck %s
// RUN: env LIBRARY_PATH=%T/test1 %clang -target i386-apple-darwin  %s -### 2>&1 | FileCheck %s
