// RUN: not %check_clang_tidy %s custom-* %t --config-file=%S/Inputs/incorrect-clang-tidy.yml

// CHECK-MESSAGES: error: unsupported query kind [clang-tidy-config]

static int S;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: find static variable [custom-test-let-bind-valid]
