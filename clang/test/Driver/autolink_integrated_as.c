// RUN: %clang -target x86_64-apple-darwin -fsyntax-only %s -no-integrated-as -### 2>&1 | FileCheck %s

// Test that the autolinking feature is disabled with *not* using the
// integrated assembler.

// CHECK: -fno-autolink
