// RUN: %check_clang_tidy -std=c11-or-later %s modernize-macro-to-enum %t-c
// RUN: %clang -std=c11 -fsyntax-only %t-c.c
// RUN: %check_clang_tidy -std=c++14-or-later -assume-filename=expand.cpp %s modernize-macro-to-enum %t-cxx
// RUN: %clang -std=c++14 -fsyntax-only %t-cxx.cpp

void use_one(int);

// A macro supplying call parentheses is not an expression. Reject its entire
// cluster, including a safely used macro and an unused macro.
#define CLUSTER_SAFE 9
#define CALL_ARGS (1)
#define CLUSTER_UNUSED 2
// CHECK-FIXES: #define CLUSTER_SAFE 9
// CHECK-FIXES-NEXT: #define CALL_ARGS (1)
// CHECK-FIXES-NEXT: #define CLUSTER_UNUSED 2

int ClusterValue = CLUSTER_SAFE;
void call_args(void) {
  use_one CALL_ARGS;
}

// A safe use cannot validate a different, unsafe use of the same macro.
#define SAFE_THEN_UNSAFE (3)
// CHECK-FIXES: #define SAFE_THEN_UNSAFE (3)

int SafeFirst = SAFE_THEN_UNSAFE;
void safe_then_unsafe(void) {
  use_one SAFE_THEN_UNSAFE;
}

#define UNSAFE_THEN_SAFE (4)
// CHECK-FIXES: #define UNSAFE_THEN_SAFE (4)

void unsafe_then_safe(void) {
  use_one UNSAFE_THEN_SAFE;
}
int SafeLast = UNSAFE_THEN_SAFE;

// Replacing either fragment with an enumerator would change parsing.
#define UNGROUPED 1 + 2
// CHECK-FIXES: #define UNGROUPED 1 + 2

int Product = 3 * UNGROUPED;

#define SUBTRACTION -1
// CHECK-FIXES: #define SUBTRACTION -1

int Difference = 3 SUBTRACTION;

// Independent safe clusters still convert, including repeated expansions.
#define SAFE_LITERAL 1
#define SAFE_UNARY -2
#define SAFE_BINARY 1 + 2
#define SAFE_PARENS ((4))
// CHECK-MESSAGES: :[[@LINE-4]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'SAFE_LITERAL' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'SAFE_UNARY' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'SAFE_BINARY' defines an integral constant; prefer an enum instead
// CHECK-MESSAGES: :[[@LINE-5]]:9: warning: macro 'SAFE_PARENS' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: SAFE_LITERAL = 1,
// CHECK-FIXES-NEXT: SAFE_UNARY = -2,
// CHECK-FIXES-NEXT: SAFE_BINARY = 1 + 2,
// CHECK-FIXES-NEXT: SAFE_PARENS = ((4))
// CHECK-FIXES-NEXT: };

int Literal = SAFE_LITERAL;
int Unary = SAFE_UNARY;
int Binary = SAFE_BINARY;
int Parens = SAFE_PARENS;

void safe_uses(void) {
  use_one(SAFE_LITERAL);
  use_one(SAFE_LITERAL);
  use_one(SAFE_UNARY);
  use_one(SAFE_BINARY);
  use_one(SAFE_PARENS);
}

// No observed expansions is not evidence of an unsafe use.
#define UNUSED 7
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'UNUSED' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: UNUSED = 7
// CHECK-FIXES-NEXT: };
