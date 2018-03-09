// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__launch_bounds__(128, 7) void Test2Args(void);
__launch_bounds__(128) void Test1Arg(void);

__launch_bounds__(0xffffffff) void TestMaxArg(void);
__launch_bounds__(0x100000000) void TestTooBigArg(void); // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__launch_bounds__(0x10000000000000000) void TestWayTooBigArg(void); // expected-error {{integer literal is too large to be represented in any integer type}}

__launch_bounds__(-128, 7) void TestNegArg1(void); // expected-warning {{'launch_bounds' attribute parameter 0 is negative and will be ignored}}
__launch_bounds__(128, -7) void TestNegArg2(void); // expected-warning {{'launch_bounds' attribute parameter 1 is negative and will be ignored}}

__launch_bounds__(1, 2, 3) void Test3Args(void); // expected-error {{'launch_bounds' attribute takes no more than 2 arguments}}
__launch_bounds__() void TestNoArgs(void); // expected-error {{'launch_bounds' attribute takes at least 1 argument}}

int TestNoFunction __launch_bounds__(128, 7); // expected-warning {{'launch_bounds' attribute only applies to Objective-C methods, functions, and function pointers}}

__launch_bounds__(true) void TestBool(void);
__launch_bounds__(128.0) void TestFP(void); // expected-error {{'launch_bounds' attribute requires parameter 0 to be an integer constant}}
__launch_bounds__((void*)0) void TestNullptr(void); // expected-error {{'launch_bounds' attribute requires parameter 0 to be an integer constant}}

int nonconstint = 256;
__launch_bounds__(nonconstint) void TestNonConstInt(void); // expected-error {{'launch_bounds' attribute requires parameter 0 to be an integer constant}}

const int constint = 512;
__launch_bounds__(constint) void TestConstInt(void);
__launch_bounds__(constint * 2 + 3) void TestConstIntExpr(void);

template <int a, int b> __launch_bounds__(a, b) void TestTemplate2Args(void) {}
template void TestTemplate2Args<128,7>(void);

template <int a> __launch_bounds__(a) void TestTemplate1Arg(void) {}
template void TestTemplate1Arg<128>(void);

template <class a>
__launch_bounds__(a) void TestTemplate1ArgClass(void) {} // expected-error {{'a' does not refer to a value}}
// expected-note@-2 {{declared here}}

template <int a, int b, int c>
__launch_bounds__(a + b, c + constint) void TestTemplateExpr(void) {}
template void TestTemplateExpr<128+constint, 3, 7>(void);

template <int... Args>
__launch_bounds__(Args) void TestTemplateVariadicArgs(void) {} // expected-error {{expression contains unexpanded parameter pack 'Args'}}

template <int... Args>
__launch_bounds__(1, Args) void TestTemplateVariadicArgs2(void) {} // expected-error {{expression contains unexpanded parameter pack 'Args'}}
