// REQUIRES: custom-check
// RUN: %check_clang_tidy %s custom-* %t --config-file=%S/Inputs/incorrect-clang-tidy.yml

// CHECK-MESSAGES: warning: 1:1: Matcher not found: varDeclInvalid in 'test-let-bind-invalid-2' [clang-tidy-config]
// CHECK-MESSAGES: warning: unsupported query kind 'SetOutputKind' in 'test-let-bind-invalid-1' [clang-tidy-config]

static int S;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: find static variable [custom-test-let-bind-valid]
