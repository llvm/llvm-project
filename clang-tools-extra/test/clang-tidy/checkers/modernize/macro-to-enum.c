// RUN: %check_clang_tidy %s modernize-macro-to-enum %t

// C requires enum values to fit into an int.
#define TOO_BIG1 1L
#define TOO_BIG2 1UL
#define TOO_BIG3 1LL
#define TOO_BIG4 1ULL

// C forbids comma operator in initializing expressions.
#define BAD_OP 1, 2

#define SIZE_OK1 1
#define SIZE_OK2 1U
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: replace macro with enum [modernize-macro-to-enum]
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'SIZE_OK1' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-3]]:9: warning: macro 'SIZE_OK2' defines an integral constant; prefer an enum instead
