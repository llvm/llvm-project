// RUN: %clang_cc1 -std=c++11 -fsyntax-only -triple nvptx-unknown-unknown -target-cpu sm_90 -verify %s

#include "Inputs/cuda.h"

__launch_bounds__(128, 7) void Test2Args(void);
__launch_bounds__(128) void Test1Arg(void);

__launch_bounds__(0xffffffff) void TestMaxArg(void);
__launch_bounds__(0x100000000) void TestTooBigArg(void); // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__launch_bounds__(0x10000000000000000) void TestWayTooBigArg(void); // expected-error {{integer literal is too large to be represented in any integer type}}
__launch_bounds__(1, 1, 0x10000000000000000) void TestWayTooBigArg(void); // expected-error {{integer literal is too large to be represented in any integer type}}

__launch_bounds__(-128, 7) void TestNegArg1(void); // expected-warning {{'launch_bounds' attribute parameter 0 is negative and will be ignored}}
__launch_bounds__(128, -7) void TestNegArg2(void); // expected-warning {{'launch_bounds' attribute parameter 1 is negative and will be ignored}}
__launch_bounds__(-128, 1, 7) void TestNegArg2(void); // expected-warning {{'launch_bounds' attribute parameter 0 is negative and will be ignored}}
__launch_bounds__(128, -1, 7) void TestNegArg2(void); // expected-warning {{'launch_bounds' attribute parameter 1 is negative and will be ignored}}
__launch_bounds__(128, 1, -7) void TestNegArg2(void); // expected-warning {{'launch_bounds' attribute parameter 2 is negative and will be ignored}}
// expected-warning@20 {{'launch_bounds' attribute parameter 0 is negative and will be ignored}}
// expected-warning@20 {{'launch_bounds' attribute parameter 1 is negative and will be ignored}}
__launch_bounds__(-128, -1, 7) void TestNegArg2(void);
// expected-warning@23 {{'launch_bounds' attribute parameter 0 is negative and will be ignored}}
// expected-warning@23 {{'launch_bounds' attribute parameter 2 is negative and will be ignored}}
__launch_bounds__(-128, 1, -7) void TestNegArg2(void);
// expected-warning@27 {{'launch_bounds' attribute parameter 0 is negative and will be ignored}}
// expected-warning@27 {{'launch_bounds' attribute parameter 1 is negative and will be ignored}}
// expected-warning@27 {{'launch_bounds' attribute parameter 2 is negative and will be ignored}}
__launch_bounds__(-128, -1, -7) void TestNegArg2(void);


__launch_bounds__(1, 2, 3, 4) void Test4Args(void); // expected-error {{'launch_bounds' attribute takes no more than 3 arguments}}
__launch_bounds__() void TestNoArgs(void); // expected-error {{'launch_bounds' attribute takes at least 1 argument}}

int TestNoFunction __launch_bounds__(128, 7, 13); // expected-warning {{'launch_bounds' attribute only applies to Objective-C methods, functions, and function pointers}}

__launch_bounds__(true) void TestBool(void);
__launch_bounds__(128, 1, 128.0) void TestFP(void); // expected-error {{'launch_bounds' attribute requires parameter 2 to be an integer constant}}
__launch_bounds__(128, 1, (void*)0) void TestNullptr(void); // expected-error {{'launch_bounds' attribute requires parameter 2 to be an integer constant}}

int nonconstint = 256;
__launch_bounds__(125, 1, nonconstint) void TestNonConstInt(void); // expected-error {{'launch_bounds' attribute requires parameter 2 to be an integer constant}}

const int constint = 512;
__launch_bounds__(128, 1, constint) void TestConstInt(void);
__launch_bounds__(128, 1, constint * 2 + 3) void TestConstIntExpr(void);

template <int a, int b, int c> __launch_bounds__(a, b, c) void TestTemplate2Args(void) {}
template void TestTemplate2Args<128,7, 13>(void);

template <int a, int b, int c>
__launch_bounds__(a + b, c + constint, a + b + c + constint) void TestTemplateExpr(void) {}
template void TestTemplateExpr<128+constint, 3, 7>(void);

template <int... Args>
__launch_bounds__(Args) void TestTemplateVariadicArgs(void) {} // expected-error {{expression contains unexpanded parameter pack 'Args'}}

template <int... Args>
__launch_bounds__(1, 22, Args) void TestTemplateVariadicArgs2(void) {} // expected-error {{expression contains unexpanded parameter pack 'Args'}}
