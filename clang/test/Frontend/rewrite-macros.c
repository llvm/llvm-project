// RUN: %clang_cc1 -verify -rewrite-macros -o %t %s

#define A(a,b) a ## b

// RUN: grep '12 */\*A\*/ /\*(1,2)\*/' %t
A(1,2)

// RUN: grep '/\*_Pragma("mark")\*/' %t
_Pragma("mark")

// RUN: grep "//#warning eek" %t
/* expected-warning {{eek}} */ #warning eek

// RUN: grep "//#pragma mark mark" %t
#pragma mark mark


