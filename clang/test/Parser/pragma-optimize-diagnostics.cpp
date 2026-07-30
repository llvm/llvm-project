// RUN: %clang_cc1 -fsyntax-only -verify %s

#pragma clang optimize off

#pragma clang optimize on

// Extra arguments
#pragma clang optimize on top of spaghetti  // expected-error {{unexpected extra argument 'top' to '#pragma clang optimize'}}

// Wrong argument
#pragma clang optimize something_wrong  // expected-error {{unexpected argument 'something_wrong' to '#pragma clang optimize'; expected 'on' or 'off'}}

// No argument
#pragma clang optimize // expected-error {{missing argument to '#pragma clang optimize'; expected 'on' or 'off'}}

// Check that macros can be used in the pragma
#define OFF off
#define ON on
#pragma clang optimize OFF
#pragma clang optimize ON

// Check that _Pragma can also be used to address the use case where users want
// to define optimization control macros to abstract out which compiler they are
// using.
#define OPT_OFF _Pragma("clang optimize off")
#define OPT_ON _Pragma("clang optimize on")
OPT_OFF
OPT_ON
