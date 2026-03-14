/* Test pragma message directive from
   http://msdn.microsoft.com/en-us/library/x7dkzch2.aspx */
// message: Sends a string literal to the standard output without terminating
// the compilation.
// #pragma message(messagestring)
// OR
// #pragma message messagestring
//
// RUN: %clang_cc1 -fsyntax-only -verify -Werror %s
// RUN: %clang_cc1 -fsyntax-only -verify -Werror -W#pragma-messages %s
#define STRING2(x) #x
#define STRING(x) STRING2(x)
#pragma message(":O I'm a message! " STRING(__LINE__)) // expected-warning {{:O I'm a message! 13}}
#pragma message ":O gcc accepts this! " STRING(__LINE__) // expected-warning {{:O gcc accepts this! 14}}

#pragma message(invalid) // expected-error {{expected string literal in pragma message}}

// GCC supports a similar pragma, #pragma GCC warning (which generates a warning
// message) and #pragma GCC error (which generates an error message).

#pragma GCC warning(":O I'm a message! " STRING(__LINE__)) // expected-warning {{:O I'm a message! 21}}
#pragma GCC warning ":O gcc accepts this! " STRING(__LINE__) // expected-warning {{:O gcc accepts this! 22}}

#pragma GCC error(":O I'm a message! " STRING(__LINE__)) // expected-error {{:O I'm a message! 24}}
#pragma GCC error ":O gcc accepts this! " STRING(__LINE__) // expected-error {{:O gcc accepts this! 25}}

#define COMPILE_ERROR(x) _Pragma(STRING2(GCC error(x)))
COMPILE_ERROR("Compile error at line " STRING(__LINE__) "!"); // expected-error {{Compile error at line 28!}}

#pragma message // expected-error {{pragma message requires parenthesized string}}
#pragma GCC warning("" // expected-error {{pragma warning requires parenthesized string}}
#pragma GCC error(1) // expected-error {{expected string literal in pragma error}}
