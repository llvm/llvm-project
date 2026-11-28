// RUN: %check_clang_tidy -std=c11-or-later %s modernize-macro-to-enum %t-c -- -- -Wno-macro-redefined
// RUN: %clang -std=c11 -Wno-macro-redefined -fsyntax-only %t-c.c
// RUN: %check_clang_tidy -std=c++14-or-later -assume-filename=redefined.cpp %s modernize-macro-to-enum %t-cxx -- -- -Wno-macro-redefined
// RUN: %clang -std=c++14 -Wno-macro-redefined -fsyntax-only %t-cxx.cpp

void use_one(int);

// A new definition's unsafe use must not attach to an older, excluded one.
void old_definition(void) {
#define REDEFINED (1)
  // CHECK-FIXES: #define REDEFINED (1)
}

#define REDEFINED (2)
// CHECK-FIXES: #define REDEFINED (2)

void redefined_use(void) {
  use_one REDEFINED;
}

// Replacing the second definition would leave the first macro active at the
// enumerator declaration, even though all uses are complete expressions.
#define REDEFINED_LITERAL unknown_identifier
// CHECK-FIXES: #define REDEFINED_LITERAL unknown_identifier
#define REDEFINED_LITERAL 3
// CHECK-FIXES: #define REDEFINED_LITERAL 3

int Literal = REDEFINED_LITERAL;

#define CONDITIONAL_REDEF (4)
// CHECK-FIXES: #define CONDITIONAL_REDEF (4)

int BeforeConditional = CONDITIONAL_REDEF;
#if 1
#define CONDITIONAL_REDEF (5)
// CHECK-FIXES: #define CONDITIONAL_REDEF (5)
#endif
int AfterConditional = CONDITIONAL_REDEF;

// Preserve the existing exclusion of all definitions of an undefined name.
#define UNDEFINED (6)
// CHECK-FIXES: #define UNDEFINED (6)

int BeforeUndef = UNDEFINED;
#undef UNDEFINED
#define UNDEFINED (7)
// CHECK-FIXES: #define UNDEFINED (7)

int AfterUndef = UNDEFINED;

// Redefined function-like macros still exclude names in their new bodies.
#define WRAPPER() 0
#define WRAPPED_VALUE 8
// CHECK-FIXES: #define WRAPPED_VALUE 8
#define WRAPPER() WRAPPED_VALUE

int Wrapped = WRAPPER();

// Unrelated definitions still convert.
#define INDEPENDENT 70
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'INDEPENDENT' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: INDEPENDENT = 70
// CHECK-FIXES-NEXT: };

int Independent = INDEPENDENT;
