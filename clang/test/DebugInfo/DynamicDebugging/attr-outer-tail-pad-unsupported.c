// REQUIRES: !x86-registered-target
// This test should trip to remind that tail-padding support should be added
// for targets as dynamic debugging support is expanded.
// RUN: %clang -cc1 %s -triple %itanium_abi_triple -debug-info-kind=constructor -fdynamic-debugging -o - \
// RUN:    -emit-llvm --discard-dynamic-debugging-debug-module \
// RUN: | FileCheck %s

// CHECK: "tail-pad-to-size"
int f() { return 0; }
