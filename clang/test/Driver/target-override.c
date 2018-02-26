// REQUIRES: shell
// REQUIRES: x86-registered-target

// RUN: mkdir -p %T/testbin
// RUN: [ ! -s %T/testbin/i386-clang ] || rm %T/testbin/i386-clang
// RUN: ln -s %clang %T/testbin/i386-clang

// Check if invocation of "foo-clang" adds option "-target foo".
//
// RUN: %T/testbin/i386-clang -c -no-canonical-prefixes %s -### 2>&1 | FileCheck -check-prefix CHECK-TG1 %s
// CHECK-TG1: Target: i386

// Check if invocation of "foo-clang -target bar" overrides option "-target foo".
//
// RUN: %T/testbin/i386-clang -c -no-canonical-prefixes -target x86_64 %s -### 2>&1 | FileCheck -check-prefix CHECK-TG2 %s
// CHECK-TG2: Target: x86_64
