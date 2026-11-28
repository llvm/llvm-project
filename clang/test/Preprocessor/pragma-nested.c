// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -fms-extensions -verify -verify-ignore-unexpected=note %s

#pragma clang attribute (_Pragma("clang attribute push")) // expected-error {{nested pragma is not allowed}}
#pragma clang attribute () // expected-error {{expected an attribute after '('}}

#define DEPRECATED_EMPTY
#pragma clang deprecated(DEPRECATED_EMPTY)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wdeprecated-pragma"
// expected-error@+2 {{macro 'DEPRECATED_EMPTY' has been marked as deprecated}}
// expected-error@+1 {{expected an attribute after '('}}
#pragma clang attribute (DEPRECATED_EMPTY)
#pragma clang diagnostic pop

#pragma clang __debug dump _Pragma("message(\"inner\")") // expected-error {{nested pragma is not allowed}} expected-warning {{missing argument to debug command 'dump'}}

#define OPEN (
#define INNER _Pragma OPEN "message(\"inner\")")
_Pragma(INNER "message(\"outer\")") // expected-error {{nested pragma is not allowed}} expected-warning {{outer}}
_Pragma(bad _Pragma("message(\"inner\")")) // expected-error {{_Pragma takes a parenthesized string literal}} expected-error {{nested pragma is not allowed}}
_Pragma(bad // expected-error {{_Pragma takes a parenthesized string literal}}
_Pragma("message(\"next\")") int after_next; // expected-warning {{next}}
int *use_after_next = &after_next;

#define EMPTY
_Pragma(bad EMPTY // expected-error {{_Pragma takes a parenthesized string literal}}
int keep_decl;
int *use_keep_decl = &keep_decl;

_Pragma(bad // expected-error {{_Pragma takes a parenthesized string literal}}
#pragma message("after") // expected-warning {{after}}
int keep_after_hash;
int *use_keep_after_hash = &keep_after_hash;

#define FLAG 1
#pragma push_macro("FLAG")
#undef FLAG
_Pragma(bad // expected-error {{_Pragma takes a parenthesized string literal}}
#if _Pragma("pop_macro(\"FLAG\")") FLAG
int keep_selected;
#else
int wrong_branch;
#endif
int *use_keep_selected = &keep_selected;

#ifdef __cplusplus
_Pragma("bad"_x _Pragma("message(\"inner\")")) // expected-error {{string literal with user-defined suffix cannot be used here}} expected-error {{nested pragma is not allowed}}
#define R )
_Pragma("bad"_x R // expected-error {{string literal with user-defined suffix cannot be used here}}
_Pragma("bad"_x // expected-error {{string literal with user-defined suffix cannot be used here}}
_Pragma("message(\"after_udl\")") int keep_after_udl; // expected-warning {{after_udl}}
int *use_keep_after_udl = &keep_after_udl;
#endif

#define L (
#define F() _Pragma("message(\"escaped\")")
#pragma clang attribute push(F L )) // expected-error {{expected an attribute}}
void loop_replay(void) {
#ifdef __cplusplus
#pragma clang loop unroll_count(F L )) // expected-error {{'F' is defined as a function-like macro; did you mean 'F(...)'?}}
#else
#pragma clang loop unroll_count(F L )) // expected-error {{call to undeclared function 'F'}} expected-error {{expression is not an integer constant expression}}
#endif
  for (int i = 0; i < 4; ++i)
    ;
}

#define PASS(x) x
#define DROP(x)
#define FORWARD_TO_DROP(x) DROP(x)
#pragma message("outer" PASS(_Pragma("message(\"inner\")"))) // expected-error {{nested pragma is not allowed}} expected-warning {{outer}}
#pragma message("outer" FORWARD_TO_DROP(_Pragma("message(\"inner\")"))) // expected-warning {{outer}}

_Pragma("message(_Pragma(\"message(\\\"inner\\\")\") \"outer\")") // expected-error {{nested pragma is not allowed}} expected-warning {{outer}}
__pragma(message("outer" __pragma(message("inner")))) // expected-error {{nested pragma is not allowed}} expected-warning {{outer}}

_Pragma(
#pragma message("inner") // expected-error {{nested pragma is not allowed}}
    "message(\"outer\")") // expected-warning@-2 {{outer}}

#pragma message("outer" _Pragma("message(\"inner\")" // expected-error {{nested pragma is not allowed}} expected-error {{pragma message requires parenthesized string}}

#if __pragma(message("outer" _Pragma("message(\"inner\")") // expected-error {{nested pragma is not allowed}} expected-error {{missing terminating ')' character}} expected-error {{expected value in expression}}
#endif

_Pragma("message(\"first\")") __pragma(message("second")) // expected-warning {{first}} expected-warning {{second}}
int after;
int *use = &after;
