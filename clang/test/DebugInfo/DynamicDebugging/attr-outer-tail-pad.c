// REQUIRES: target=x86_64-sie-ps5
/// Requires target because object emission occurs (FIXME: add option to not).

// RUN: %clang %s -c -O1 -Xclang -disable-llvm-passes -g -fdynamic-debugging -o - -emit-llvm -S --target=x86_64-sie-ps5 | FileCheck %s --check-prefix=PS5
/// FIXME: Add negative tests for other targets (if in doubt, don't tail-pad).

/// Pad functions to minimum of 5 bytes for insertion of 32 rel jump.
// PS5: define hidden i32 @f() #0
// PS5: attributes #0 = {{{.*}}"tail-pad-to-size"="5"{{.*}}"tail-pad-value"="144"{{.*}}}
int f() { return 0; }
