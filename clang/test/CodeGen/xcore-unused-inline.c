// REQUIRES: xcore-registered-target
// RUN: %clang_cc1 -triple xcore-unknown-unknown -emit-llvm -o - %s

// D77068 fixes a segmentation fault and assertion failure "Unexpected null
// Value" in the case of an unused inline function, when targeting xcore. This
// test verifies that clang does not crash and does not produce code for such a
// function.

inline void dead_function(void) {}
