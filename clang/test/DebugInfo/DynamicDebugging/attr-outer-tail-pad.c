// REQUIRES: x86-registered-target
// RUN: %clang -cc1 %s -triple x86_64-unknown-unknown -debug-info-kind=constructor -fdynamic-debugging -o - \
// RUN:    -emit-llvm --discard-dynamic-debugging-debug-module \
// RUN: | FileCheck %s --check-prefix=X86

/// Pad functions to minimum of 5 bytes for insertion of 32 rel jump.
// X86: define dso_local i32 @f() #0
// X86: attributes #0 =
// X86-SAME: "tail-pad-to-size"="5"
// X86-SAME: "tail-pad-value"="144"
int f() { return 0; }
