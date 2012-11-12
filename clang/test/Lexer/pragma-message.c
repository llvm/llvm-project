/* Test pragma message directive from
   http://msdn.microsoft.com/en-us/library/x7dkzch2.aspx */

// message: Sends a string literal to the standard output without terminating
// the compilation.
// #pragma message(messagestring)
// OR
// #pragma message messagestring
//
// RUN: %clang_cc1 -fsyntax-only -verify -Werror %s
#define STRING2(x) #x
#define STRING(x) STRING2(x)
#pragma message(":O I'm a message! " STRING(__LINE__)) // expected-warning {{:O I'm a message! 13}}
#pragma message ":O gcc accepts this! " STRING(__LINE__) // expected-warning {{:O gcc accepts this! 14}}
