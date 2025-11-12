// REQUIRES: custom-check
// RUN: %check_clang_tidy %s custom-* %t --config-file=%S/Inputs/clang-tidy.yml

extern long E;
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: use 'int' instead of 'long' [custom-test-diag-level]
// CHECK-MESSAGES: [[@LINE-2]]:1: note: declaration of 'long'
static int S;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: find static variable [custom-test-let-bind]
