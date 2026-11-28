// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-macro-to-enum %t -- -- -Wno-unused-value -fno-delayed-template-parsing
// RUN: %clang -std=c++14 -Wno-unused-value -fno-delayed-template-parsing -fsyntax-only %t.cpp

void use_one(int);
void use_two(int, int);

// Validate the defining macro through several object-like expansions.
#define NESTED_VALUE 12345
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'NESTED_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: NESTED_VALUE = 12345
// CHECK-FIXES-NEXT: };
#define VALUE_ALIAS NESTED_VALUE
#define SECOND_ALIAS VALUE_ALIAS

int Nested = SECOND_ALIAS;

// The inner expression need not consume its enclosing macro expansion.
#define NESTED_TERM (12 + 34)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'NESTED_TERM' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: NESTED_TERM = (12 + 34)
// CHECK-FIXES-NEXT: };
#define TERM_ALIAS 3 * NESTED_TERM

int Term = TERM_ALIAS;

// A containing expression must not validate a partial inner expansion.
#define PARTIAL_VALUE 1 + 2
// CHECK-FIXES: #define PARTIAL_VALUE 1 + 2
#define PARTIAL_ALIAS 3 * PARTIAL_VALUE

int Partial = PARTIAL_ALIAS;

// One safe nested use cannot validate another invocation used as syntax.
#define MIXED_ARGS (1, 2)
// CHECK-FIXES: #define MIXED_ARGS (1, 2)
#define ARGS_ALIAS MIXED_ARGS

int Mixed = ARGS_ALIAS;
void nested_call() {
  use_two ARGS_ALIAS;
}

// Preserve exclusions for references in function-like macro bodies.
#define BODY_VALUE 20
// CHECK-FIXES: #define BODY_VALUE 20
#define BODY_WRAPPER() BODY_VALUE

int Body = BODY_WRAPPER();

#define ARG_VALUE 21
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'ARG_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: ARG_VALUE = 21
// CHECK-FIXES-NEXT: };
#define IDENTITY(x) x

int Argument = IDENTITY(ARG_VALUE);

#define PASTE_VALUE 22
// CHECK-FIXES: #define PASTE_VALUE 22
#define PASTE_IMPL(a, b) a##b
#define PASTE(a, b) PASTE_IMPL(a, b)

int pasted22;
int *Pasted = &PASTE(pasted, PASTE_VALUE);

// Instantiations revisit source locations already seen in the template body.
#define TEMPLATE_VALUE (31)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'TEMPLATE_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: TEMPLATE_VALUE = (31)
// CHECK-FIXES-NEXT: };

template <typename T> int safe_template(T) {
  use_one(TEMPLATE_VALUE);
  return TEMPLATE_VALUE;
}

int First = safe_template(0);
int Second = safe_template(0L);

#define TEMPLATE_ARGS (32)
// CHECK-FIXES: #define TEMPLATE_ARGS (32)

template <typename T> void unsafe_template(T) {
  use_one TEMPLATE_ARGS;
}

void instantiate() {
  unsafe_template(0);
  unsafe_template(0L);
}
