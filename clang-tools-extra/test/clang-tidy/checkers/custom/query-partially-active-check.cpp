// REQUIRES: custom-check
// RUN: %check_clang_tidy %s custom-test-let-bind %t --config-file=%S/Inputs/clang-tidy.yml

extern long E;
static int S;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: find static variable [custom-test-let-bind]
