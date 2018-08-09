// RUN: rm -rf %t
// RUN: mkdir %t
//
// RUN: env LIBRARY_PATH=%t/test1 %clang -x c %s -### 2>&1 | FileCheck %s
// CHECK: "-L{{.*}}/test1"

// GCC driver is used as linker on cygming. It should be aware of LIBRARY_PATH.
// XFAIL: windows-msvc
// REQUIRES: clang-driver
// REQUIRES: native

// Make sure that LIBRARY_PATH works for both i386 and x86_64 on Darwin.
// RUN: env LIBRARY_PATH=%t/test1 %clang -target x86_64-apple-darwin %s -### 2>&1 | FileCheck %s
// RUN: env LIBRARY_PATH=%t/test1 %clang -target i386-apple-darwin  %s -### 2>&1 | FileCheck %s
//
// Make sure that we don't warn on unused compiler arguments.
// RUN: %clang -Xclang -I. -x c %s -c -o %t/tmp.o
// RUN: %clang -Xclang -I. %t/tmp.o -o %t/tmp -### 2>&1 | FileCheck %s --check-prefix=NO-UNUSED
// NO-UNUSED-NOT: warning:{{.*}}unused
//
// Make sure that we do warn in other cases.
// RUN: %clang %s -lfoo -c -o %t/tmp2.o -### 2>&1 | FileCheck %s --check-prefix=UNUSED
// UNUSED: warning:{{.*}}unused
