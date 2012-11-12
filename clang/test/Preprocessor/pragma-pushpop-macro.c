/* Test pragma pop_macro and push_macro directives from
   http://msdn.microsoft.com/en-us/library/hsttss76.aspx */

// pop_macro: Sets the value of the macro_name macro to the value on the top of
// the stack for this macro.
// #pragma pop_macro("macro_name")
// push_macro: Saves the value of the macro_name macro on the top of the stack
// for this macro.
// #pragma push_macro("macro_name")
//
// RUN: %clang_cc1 -fms-extensions -E %s -o - | FileCheck %s

#define X 1
#define Y 2
int pmx0 = X;
int pmy0 = Y;
#define Y 3
#pragma push_macro("Y")
#pragma push_macro("X")
int pmx1 = X;
#define X 2
int pmx2 = X;
#pragma pop_macro("X")
int pmx3 = X;
#pragma pop_macro("Y")
int pmy1 = Y;

// Have a stray 'push' to show we don't crash when having imbalanced
// push/pop
#pragma push_macro("Y")
#define Y 4
int pmy2 = Y;

// The sequence push, define/undef, pop caused problems if macro was not
// previously defined.
#pragma push_macro("PREVIOUSLY_UNDEFINED1")
#undef PREVIOUSLY_UNDEFINED1
#pragma pop_macro("PREVIOUSLY_UNDEFINED1")
#ifndef PREVIOUSLY_UNDEFINED1
int Q;
#endif

#pragma push_macro("PREVIOUSLY_UNDEFINED2")
#define PREVIOUSLY_UNDEFINED2
#pragma pop_macro("PREVIOUSLY_UNDEFINED2")
#ifndef PREVIOUSLY_UNDEFINED2
int P;
#endif

// CHECK: int pmx0 = 1
// CHECK: int pmy0 = 2
// CHECK: int pmx1 = 1
// CHECK: int pmx2 = 2
// CHECK: int pmx3 = 1
// CHECK: int pmy1 = 3
// CHECK: int pmy2 = 4
// CHECK: int Q;
// CHECK: int P;
