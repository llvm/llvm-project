// RUN: %clang_cc1 %s -verify -rewrite-macros -o %t
// RUN: FileCheck %s < %t

// Any CHECK line comments are included in the output, so we use some extra
// regex brackets to make sure we don't match the CHECK lines themselves.

#define A(a,b) a ## b

// CHECK: {{^}} 12 /*A*/ /*(1,2)*/{{$}}
A(1,2)

// CHECK: {{^}} /*_Pragma("mark")*/{{$}}
_Pragma("mark")

// CHECK: /*#warning eek*/{{$}}
/* expected-warning {{eek}} */ #warning eek

// CHECK: {{^}}//#pragma mark mark{{$}}
#pragma mark mark
