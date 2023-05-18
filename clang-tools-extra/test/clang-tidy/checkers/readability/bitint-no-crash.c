// RUN: %check_clang_tidy %s readability-magic-numbers %t --

// Don't crash

_BitInt(128) A = 4533629751480627964421wb;
// CHECK-MESSAGES: warning
