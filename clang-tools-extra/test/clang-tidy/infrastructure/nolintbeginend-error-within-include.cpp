// RUN: %check_clang_tidy %s google-explicit-constructor %t -- --header-filter=.* -system-headers -- -isystem %S/Inputs/nolintbeginend

#include "error_in_include.inc"
// CHECK-MESSAGES: error_in_include.inc:1:11: warning: single-argument constructors must be marked explicit

#include "nolint_in_include.inc"

// CHECK-MESSAGES: Suppressed 1 warnings (1 NOLINT).
