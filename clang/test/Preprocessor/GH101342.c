// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -pedantic -verify

// GH101342
// We should not form identifiers by concatenating invalid UCNs.

#define DOT •
#define CONCAT_IMPL(Left, Separator, Right) Left##Separator##Right
#define CONCAT(Left, Separator, Right) CONCAT_IMPL(Left, Separator, Right)
#define MAKE_CLASS_NAME(A, B) CONCAT(A, DOT, B)

// struct foo•bar {} x;
struct MAKE_CLASS_NAME(foo, bar);
// expected-error@-1 {{pasting formed 'foo•', an invalid preprocessing token}} \
// expected-error@-1 {{pasting formed '•bar', an invalid preprocessing token}}
#ifdef __cplusplus
// expected-error@-4 {{expected unqualified-id}}
#else
// expected-error@-6 {{expected identifier or '('}}
#endif
