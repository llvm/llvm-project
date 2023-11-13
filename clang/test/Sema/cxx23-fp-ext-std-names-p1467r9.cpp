// RUN: %clang_cc1 -fsyntax-only -std=c++23 -target-feature +fullbf16 -verify -ast-dump %s | FileCheck %s
#include <stdarg.h>
_Float16 f16_val_1 = 1.0bf16; // expected-error {{cannot initialize a variable of type '_Float16' with an rvalue of type '__bf16'}}
_Float16 f16_val_2 = 1.0f; // expected-error {{cannot initialize a variable of type '_Float16' with an rvalue of type 'float'}}
_Float16 f16_val_3 = 1.0; // expected-error {{cannot initialize a variable of type '_Float16' with an rvalue of type 'double'}}
_Float16 f16_val_4 = 1.0l; // expected-error {{cannot initialize a variable of type '_Float16' with an rvalue of type 'long double'}}
_Float16 f16_val_6 = 1.0f16;
//CHECK:      VarDecl {{.*}} f16_val_6 '_Float16' cinit
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
_Float16 f16_val_7 = static_cast<_Float16>(1.0bf16); // expected-error {{static_cast from '__bf16' to '_Float16' is not allowed}}
_Float16 f16_val_8 = static_cast<_Float16>(1.0f);
//CHECK:      VarDecl {{.*}} f16_val_8 '_Float16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} '_Float16' static_cast<_Float16> <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
_Float16 f16_val_9 = static_cast<_Float16>(1.0);
//CHECK:      VarDecl {{.*}} f16_val_9 '_Float16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} '_Float16' static_cast<_Float16> <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
_Float16 f16_val_10 = static_cast<_Float16>(1.0l);
//CHECK:      VarDecl {{.*}} f16_val_10 '_Float16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} '_Float16' static_cast<_Float16> <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'long double' 1.000000e+00
_Float16 f16_val_11 = static_cast<_Float16>(1.0f16);
//CHECK:      VarDecl {{.*}} f16_val_11 '_Float16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} '_Float16' static_cast<_Float16> <NoOp>
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00

decltype(0.0BF16) bf16_val_1 = 1.0f16; // expected-error {{cannot initialize a variable of type 'decltype(0.BF16)' (aka '__bf16') with an rvalue of type '_Float16'}}
decltype(0.0BF16) bf16_val_2 = 1.0f; // expected-error {{cannot initialize a variable of type 'decltype(0.BF16)' (aka '__bf16') with an rvalue of type 'float'}}
decltype(0.0BF16) bf16_val_3 = 1.0; // expected-error {{cannot initialize a variable of type 'decltype(0.BF16)' (aka '__bf16') with an rvalue of type 'double'}}
decltype(0.0BF16) bf16_val_4 = 1.0l; // expected-error {{cannot initialize a variable of type 'decltype(0.BF16)' (aka '__bf16') with an rvalue of type 'long double'}}
decltype(0.0BF16) bf16_val_5 = 1.0bf16;
//CHECK:      VarDecl {{.*}} bf16_val_5 'decltype(0.BF16)':'__bf16' cinit
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00

decltype(0.0BF16) bf16_val_6 = static_cast<decltype(0.0BF16)>(1.0f16); // expected-error {{static_cast from '_Float16' to 'decltype(0.BF16)' (aka '__bf16') is not allowed}}
decltype(0.0BF16) bf16_val_7 = static_cast<decltype(0.0BF16)>(1.0f);
//CHECK:      VarDecl {{.*}} bf16_val_7 'decltype(0.BF16)':'__bf16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' static_cast<decltype(0.BF16)> <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
decltype(0.0BF16) bf16_val_8 = static_cast<decltype(0.0BF16)>(1.0);
//CHECK:      VarDecl {{.*}} bf16_val_8 'decltype(0.BF16)':'__bf16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' static_cast<decltype(0.BF16)> <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
decltype(0.0BF16) bf16_val_9 = static_cast<decltype(0.0BF16)>(1.0l);
//CHECK:      VarDecl {{.*}} bf16_val_9 'decltype(0.BF16)':'__bf16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' static_cast<decltype(0.BF16)> <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'long double' 1.000000e+00
decltype(0.0BF16) bf16_val_10 = static_cast<decltype(0.0BF16)>(1.0bf16);
//CHECK:      VarDecl {{.*}} bf16_val_10 'decltype(0.BF16)':'__bf16' cinit
//CHECK-NEXT: CXXStaticCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' static_cast<decltype(0.BF16)> <NoOp>
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00

float f_val_1 = 1.0f16;
//CHECK:      VarDecl {{.*}} f_val_1 'float' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
float f_val_2 = 1.0bf16;
//CHECK:      VarDecl {{.*}} f_val_2 'float' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
float f_val_3 = 1.0;
//CHECK:      VarDecl {{.*}} f_val_3 'float' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
float f_val_4 = 1.0l;
//CHECK:      VarDecl {{.*}} f_val_4 'float' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'long double' 1.000000e+00
float f_val_5 = 1.0f;
//CHECK:      VarDecl {{.*}} f_val_5 'float' cinit
//CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00

double d_val_1 = 1.0f16;
//CHECK:      VarDecl {{.*}} d_val_1 'double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
double d_val_2 = 1.0bf16;
//CHECK:      VarDecl {{.*}} d_val_2 'double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
double d_val_3 = 1.0f;
//CHECK:      VarDecl {{.*}} d_val_3 'double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
double d_val_4 = 1.0l;
//CHECK:      VarDecl {{.*}} d_val_4 'double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'long double' 1.000000e+00
double d_val_5 = 1.0;
//CHECK:      VarDecl {{.*}} d_val_5 'double' cinit
//CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00

long double ld_val_1 = 1.0f16;
//CHECK:      VarDecl {{.*}} ld_val_1 'long double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
long double ld_val_2 = 1.0bf16;
//CHECK:      VarDecl {{.*}} ld_val_2 'long double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
long double ld_val_3 = 1.0f;
//CHECK:      VarDecl {{.*}} ld_val_3 'long double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
long double ld_val_4 = 1.0;
//CHECK:      VarDecl {{.*}} ld_val_4 'long double' cinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
long double ld_val_5 = 1.0l;
//CHECK:      VarDecl {{.*}} ld_val_5 'long double' cinit
//CHECK-NEXT: FloatingLiteral {{.*}} 'long double' 1.000000e+00

auto f16_bf16 = 1.0f16 + 1.0bf16; // expected-error {{invalid operands to binary expression ('_Float16' and '__bf16')}}
auto f16_bf16_cast = 1.0f16 + static_cast<_Float16>(1.0bf16); // expected-error {{static_cast from '__bf16' to '_Float16' is not allowed}}
auto f16_float = 1.0f16 + 1.0f;
//CHECK:      VarDecl {{.*}} f16_float 'float' cinit
//CHECK-NEXT: BinaryOperator {{.*}} 'float' '+'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
auto f16_double = 1.0f16 + 1.0;
//CHECK:      VarDecl {{.*}} f16_double 'double' cinit
//CHECK-NEXT: BinaryOperator {{.*}} 'double' '+'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
auto f16_ldouble = 1.0f16 + 1.0l;
//CHECK:      VarDecl {{.*}} f16_ldouble 'long double' cinit
//CHECK-NEXT: BinaryOperator {{.*}} 'long double' '+'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: FloatingLiteral {{.*}} 'long double' 1.000000e+00
auto f16_int = 1.0f16 + 1;
//CHECK:      VarDecl {{.*}} f16_int '_Float16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
auto f16_uint = 1.0f16 + 1u;
//CHECK:      VarDecl {{.*}} f16_uint '_Float16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
auto f16_long = 1.0f16 + 1l;
//CHECK:      VarDecl {{.*}} f16_long '_Float16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'long' 1
auto f16_ulong = 1.0f16 + 1ul;
//CHECK:      VarDecl {{.*}} f16_ulong '_Float16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
auto f16_llong = 1.0f16 + 1ll;
//CHECK:      VarDecl {{.*}} f16_llong '_Float16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'long long' 1
auto f16_ullong = 1.0f16 + 1ull;
//CHECK:      VarDecl {{.*}} f16_ullong '_Float16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long long' 1
auto f16_bool = 1.0f16 + true;
//CHECK:      VarDecl {{.*}} f16_bool '_Float16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '_Float16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <IntegralToFloating>
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <IntegralCast>
//CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true

auto bf16_fp16 = 1.0bf16 + 1.0f16; // expected-error {{invalid operands to binary expression ('__bf16' and '_Float16')}}
auto bf16_fp16_cast = 1.0bf16 + static_cast<decltype(0.0BF16)>(1.0f16); // expected-error {{static_cast from '_Float16' to 'decltype(0.BF16)' (aka '__bf16') is not allowed}}
auto bf16_float = 1.0bf16 + 1.0f;
//CHECK:      VarDecl {{.*}} bf16_float 'float' cinit
//CHECK-NEXT: BinaryOperator {{.*}} 'float' '+'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
auto bf16_double = 1.0bf16 + 1.0;
//CHECK:      VarDecl {{.*}} bf16_double 'double' cinit
//CHECK-NEXT: BinaryOperator {{.*}} 'double' '+'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.000000e+00
auto bf16_ldouble = 1.0bf16 + 1.0l;
//CHECK:      VarDecl {{.*}} bf16_ldouble 'long double' cinit
//CHECK-NEXT: BinaryOperator {{.*}} 'long double' '+'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: FloatingLiteral {{.*}} 'long double' 1.000000e+00
auto bf16_int = 1.0bf16 + 1;
//CHECK:      VarDecl {{.*}} bf16_int '__bf16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '__bf16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '__bf16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
auto bf16_uint = 1.0bf16 + 1u;
//CHECK:      VarDecl {{.*}} bf16_uint '__bf16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '__bf16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '__bf16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
auto bf16_long = 1.0bf16 + 1l;
//CHECK:      VarDecl {{.*}} bf16_long '__bf16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '__bf16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '__bf16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'long' 1
auto bf16_ulong = 1.0bf16 + 1ul;
//CHECK:      VarDecl {{.*}} bf16_ulong '__bf16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '__bf16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '__bf16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
auto bf16_llong = 1.0bf16 + 1ll;
//CHECK:      VarDecl {{.*}} bf16_llong '__bf16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '__bf16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '__bf16' <IntegralToFloating>
//CHECK-NEXT:  IntegerLiteral {{.*}} 'long long' 1
auto bf16_ullong = 1.0bf16 + 1ull;
//CHECK:      VarDecl {{.*}} bf16_ullong '__bf16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '__bf16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '__bf16' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long long' 1
auto bf16_bool = 1.0bf16 + true;
//CHECK:      VarDecl {{.*}} bf16_bool '__bf16' cinit
//CHECK-NEXT: BinaryOperator {{.*}} '__bf16' '+'
//CHECK-NEXT: FloatingLiteral {{.*}} '__bf16' 1.000000e+00
//CHECK-NEXT: ImplicitCastExpr {{.*}} '__bf16' <IntegralToFloating>
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <IntegralCast>
//CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true

int f(decltype(0.0BF16)) {}
int f(_Float16) {}
int f(float) {}
int f(double) {}
int f(long double) {}
int f(int) {}

decltype(0.0BF16) bf16_val = 1.0bf16;
_Float16 float16_val = 1.0f16;
float float_val = 1.0f;
double double_val = 1.0;
long double long_double_val = 1.0l;
int int_val = 1;

int test1 = f(bf16_val); // calls f(decltype(0.BF16))
//CHECK:      VarDecl {{.*}} test1 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(decltype(0.BF16))' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (decltype(0.BF16))' lvalue Function {{.*}} 'f' 'int (decltype(0.BF16))'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'decltype(0.BF16)':'__bf16' lvalue Var {{.*}} 'bf16_val' 'decltype(0.BF16)':'__bf16'
int test2 = f(float16_val); // calls f(_Float16)
//CHECK:      VarDecl {{.*}} test2 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(_Float16)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (_Float16)' lvalue Function {{.*}} 'f' 'int (_Float16)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'float16_val' '_Float16'
int test3 = f(float_val); // calls f(float)
//CHECK:      VarDecl {{.*}} test3 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(float)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (float)' lvalue Function {{.*}} 'f' 'int (float)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue Var {{.*}} 'float_val' 'float'
int test4 = f(double_val); // calls f(double)
//CHECK:      VarDecl {{.*}} test4 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(double)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (double)' lvalue Function {{.*}} 'f' 'int (double)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'double' lvalue Var {{.*}} 'double_val' 'double'
int test5 = f(long_double_val); // calls f(long double)
//CHECK:      VarDecl {{.*}} test5 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(long double)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (long double)' lvalue Function {{.*}} 'f' 'int (long double)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'long double' lvalue Var {{.*}} 'long_double_val' 'long double'
int test6 = f(int_val); // calls f(int)
//CHECK:      VarDecl {{.*}} test6 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(int)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (int)' lvalue Function {{.*}} 'f' 'int (int)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'int_val' 'int'

int f_1(float) {} // expected-note {{candidate function}} expected-note {{candidate function}}
int f_1(double) {} // expected-note {{candidate function}} expected-note {{candidate function}}

// Ambiguous cases
int test_7 = f_1(bf16_val); // expected-error {{call to 'f_1' is ambiguous}}
int test_8 = f_1(float16_val); // expected-error {{call to 'f_1' is ambiguous}}

int f_2(long double) {}
int test_9 = f_2(float_val);
//CHECK:      VarDecl {{.*}} test_9 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(long double)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (long double)' lvalue Function {{.*}} 'f_2' 'int (long double)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue Var {{.*}} 'float_val' 'float'
int test_10 = f_2(double_val);
//CHECK:      VarDecl {{.*}} test_10 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(long double)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (long double)' lvalue Function {{.*}} 'f_2' 'int (long double)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'double' lvalue Var {{.*}} 'double_val' 'double'
int test_11 = f_2(bf16_val);
//CHECK:      VarDecl {{.*}} test_11 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(long double)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (long double)' lvalue Function {{.*}} 'f_2' 'int (long double)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'decltype(0.BF16)':'__bf16' lvalue Var {{.*}} 'bf16_val' 'decltype(0.BF16)':'__bf16'
int test_12 = f_2(float16_val);
//CHECK:      VarDecl {{.*}} test_12 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(long double)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (long double)' lvalue Function {{.*}} 'f_2' 'int (long double)'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast>
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'float16_val' '_Float16'

int f_3(_Float16) {} // expected-note {{candidate function not viable: no known conversion from 'float' to '_Float16' for 1st argument}} expected-note {{no known conversion from 'decltype(0.BF16)' (aka '__bf16') to '_Float16' for 1st argument}}
int test_13 = f_3(float_val); // expected-error {{no matching function for call to 'f_3'}}
int test_14 = f_3(bf16_val); // expected-error {{no matching function for call to 'f_3'}}
int test_15 = f_3(static_cast<_Float16>(bf16_val)); // expected-error {{static_cast from 'decltype(0.BF16)' (aka '__bf16') to '_Float16' is not allowed}}

int f_4(decltype(0.0BF16)) {} // expected-note {{candidate function not viable: no known conversion from 'float' to 'decltype(0.BF16)' (aka '__bf16') for 1st argument}} expected-note {{candidate function not viable: no known conversion from '_Float16' to 'decltype(0.BF16)' (aka '__bf16') for 1st argument}}
int test_16 = f_4(float_val); // expected-error {{no matching function for call to 'f_4'}}
int test_17 = f_4(float16_val); // expected-error {{no matching function for call to 'f_4'}}
int test_18 = f_4(static_cast<decltype(0.0BF16)>(float16_val)); // expected-error {{static_cast from '_Float16' to 'decltype(0.BF16)' (aka '__bf16') is not allowed}}

struct S {
  operator decltype(0.0BF16)() const {
      return 0.0bf16;
  }
  operator _Float16() const {
      return 0.0f16;
  }
  operator float() const {
      return 0.0f;
  }
  operator double() const {
      return 0.0;
  }
  operator long double() const {
      return 0.0L;
  }
  operator int() const {
      return 0;
  }
};


S user_defined_val;
// User-defined overload cases
decltype(0.0BF16) bfloat16_val(user_defined_val); // calls operator decltype(0.BF16)()
//CHECK:      VarDecl {{.*}} bfloat16_val 'decltype(0.BF16)':'__bf16' callinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' <UserDefinedConversion>
//CHECK-NEXT: CXXMemberCallExpr {{.*}} 'decltype(0.BF16)':'__bf16'
//CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator __bf16 {{.*}}
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
//CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'user_defined_val' 'S'
_Float16 f16_val(user_defined_val); // calls operator _Float16()
//CHECK:      VarDecl {{.*}} f16_val '_Float16' callinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <UserDefinedConversion>
//CHECK-NEXT: CXXMemberCallExpr {{.*}} '_Float16
//CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator _Float16 {{.*}}
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
//CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'user_defined_val' 'S'
float f_val(user_defined_val); // calls operator float()
//CHECK:      VarDecl {{.*}} f_val 'float' callinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <UserDefinedConversion>
//CHECK-NEXT: CXXMemberCallExpr {{.*}} 'float'
//CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator float {{.*}}
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
//CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'user_defined_val' 'S'
double d_val(user_defined_val); // calls operator double()
//CHECK:      VarDecl {{.*}} d_val 'double' callinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <UserDefinedConversion>
//CHECK-NEXT: CXXMemberCallExpr {{.*}} 'double'
//CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator double {{.*}}
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
//CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'user_defined_val' 'S'
long double ld_val(user_defined_val); // calls operator long double()
//CHECK:      VarDecl {{.*}} ld_val 'long double' callinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <UserDefinedConversion>
//CHECK-NEXT: CXXMemberCallExpr {{.*}} 'long double'
//CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator long double {{.*}}
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
//CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'user_defined_val' 'S'
int i_val(user_defined_val); // calls operator int()
//CHECK:      VarDecl {{.*}} i_val 'int' callinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <UserDefinedConversion>
//CHECK-NEXT: CXXMemberCallExpr {{.*}} 'int'
//CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator int {{.*}}
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
//CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'user_defined_val' 'S'
struct S1 {
  operator _Float16() const {
      return 0.0f16;
  }
  operator float() const {
      return 0.0f;
  }
  operator double() const {
      return 0.0;
  }
  operator long double() const {
      return 0.0L;
  }
  operator int() const {
      return 0;
  }
};

S1 user_defined_val_2;
// User-defined overload cases
decltype(0.0BF16) bfloat16_val_2(user_defined_val_2); // calls operator int()
//CHECK:      VarDecl {{.*}} bfloat16_val_2 'decltype(0.BF16)':'__bf16' callinit
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' <IntegralToFloating>
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <UserDefinedConversion>
//CHECK-NEXT: CXXMemberCallExpr {{.*}} 'int'
//CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator int {{.*}}
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'const S1' lvalue <NoOp>
//CHECK-NEXT: DeclRefExpr {{.*}} 'S1' lvalue Var {{.*}} 'user_defined_val_2' 'S1'
struct S2 {
  operator decltype(0.0BF16)() const { // expected-note {{candidate function}}
      return 0.0bf16;
  }

  operator _Float16() const { // expected-note {{candidate function}}
      return 0.0f16;
  }
  operator double() const { // expected-note {{candidate function}}
      return 0.0;
  }
  operator long double() const { // expected-note {{candidate function}}
      return 0.0L;
  }
  operator int() const { // expected-note {{candidate function}}
      return 0;
  }
};

S2 user_defined_val_3;
// User-defined overload cases
float float_val_2(user_defined_val_3); // expected-error {{conversion from 'S2' to 'float' is ambiguous}}

// Test case for varadic function
int f_5(int a, ...) {
    va_list ap;
    va_start(ap, a);
    auto bf16_val = va_arg(ap, decltype(0.BF16));
    auto float16_val = va_arg(ap, _Float16);
    auto promoted_float_val = va_arg(ap, double);
    auto double_val = va_arg(ap, double);
    auto long_double_val = va_arg(ap, long double);
    auto int_val = va_arg(ap, int);
    va_end(ap);
}
// CHECK: VarDecl {{.*}} used ap 'va_list'
// CHECK: VarDecl {{.*}} bf16_val 'decltype(0.BF16)'
// CHECK: VarDecl {{.*}} float16_val '_Float16'
// CHECK: VarDecl {{.*}} promoted_float_val 'double'
// CHECK: VarDecl {{.*}} double_val 'double'
// CHECK: VarDecl {{.*}} long_double_val 'long double'
// CHECK: VarDecl {{.*}} int_val 'int'

int test_19 = f_5(0, bf16_val, float16_val, float_val, double_val, long_double_val, int_val);
//CHECK:      VarDecl {{.*}} test_19 'int' cinit
//CHECK-NEXT: CallExpr {{.*}} 'int'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(int, ...)' <FunctionToPointerDecay>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int (int, ...)' lvalue Function {{.*}} 'f_5' 'int (int, ...)'
//CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'decltype(0.BF16)':'__bf16' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'decltype(0.BF16)':'__bf16' lvalue Var {{.*}} 'bf16_val' 'decltype(0.BF16)':'__bf16'
//CHECK-NEXT: ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'float16_val' '_Float16'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue Var {{.*}} 'float_val' 'float'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'double' lvalue Var {{.*}} 'double_val' 'double'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'long double' lvalue Var {{.*}} 'long_double_val' 'long double'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
//CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'int_val' 'int'
