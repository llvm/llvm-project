// Test this without pch.
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -include %S/cxx_exprs.h -std=c++11 -fsyntax-only -verify %s -ast-dump | FileCheck %s

// Test with pch. Use '-ast-dump' to force deserialization of function bodies.
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -x c++-header -std=c++11 -emit-pch -o %t %S/cxx_exprs.h
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-dump-all | FileCheck %s

// expected-no-diagnostics

int integer;
double floating;
char character;
bool boolean;

// CXXStaticCastExpr
static_cast_result void_ptr = &integer;
// CHECK: TypedefDecl {{.*}} <{{.*}}, col:{{.*}}> col:{{.*}}{{(imported)?}} referenced static_cast_result 'typeof (static_cast<void *>(0))':'void *'{{$}}
// CHECK-NEXT: TypeOfExprType {{.*}} 'typeof (static_cast<void *>(0))' sugar{{( imported)?}}{{$}}
// CHECK-NEXT: ParenExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'void *'{{$}}
// CHECK-NEXT: CXXStaticCastExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'void *' static_cast<void *> <NoOp>{{$}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'void *' <NullToPointer> part_of_explicit_cast{{$}}
// CHECK-NEXT: IntegerLiteral {{.*}} <col:{{.*}}> 'int' 0{{$}}

// CXXDynamicCastExpr
Derived *d;
dynamic_cast_result derived_ptr = d;
// CHECK: TypedefDecl {{.*}} <{{.*}}, col:{{.*}}> col:{{.*}} referenced dynamic_cast_result 'typeof (dynamic_cast<Derived *>(base_ptr))':'Derived *'{{$}}
// CHECK-NEXT: TypeOfExprType {{.*}} 'typeof (dynamic_cast<Derived *>(base_ptr))' sugar{{( imported)?}}{{$}}
// CHECK-NEXT: ParenExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'Derived *'{{$}}
// CHECK-NEXT: CXXDynamicCastExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'Derived *' dynamic_cast<struct Derived *> <Dynamic>{{$}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'Base *' <LValueToRValue> part_of_explicit_cast{{$}}
// CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'Base *' lvalue Var {{.*}} 'base_ptr' 'Base *'{{$}}

// CXXReinterpretCastExpr
reinterpret_cast_result void_ptr2 = &integer;
// CHECK: TypedefDecl {{.*}} <{{.*}}, col:{{.*}}> col:{{.*}} referenced reinterpret_cast_result 'typeof (reinterpret_cast<void *>(0))':'void *'{{$}}
// CHECK-NEXT: TypeOfExprType {{.*}} 'typeof (reinterpret_cast<void *>(0))' sugar{{( imported)?}}{{$}}
// CHECK-NEXT: ParenExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'void *'{{$}}
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'void *' reinterpret_cast<void *> <IntegralToPointer>{{$}}
// CHECK-NEXT: IntegerLiteral {{.*}} <col:{{.*}}> 'int' 0{{$}}

// CXXConstCastExpr
const_cast_result char_ptr = &character;
// CHECK: TypedefDecl {{.*}} <{{.*}}, col:{{.*}}> col:{{.*}} referenced const_cast_result 'typeof (const_cast<char *>(const_char_ptr_value))':'char *'{{$}}
// CHECK-NEXT: TypeOfExprType {{.*}} 'typeof (const_cast<char *>(const_char_ptr_value))' sugar{{( imported)?}}{{$}}
// CHECK-NEXT: ParenExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'char *'{{$}}
// CHECK-NEXT: CXXConstCastExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'char *' const_cast<char *> <NoOp>{{$}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'const char *' <LValueToRValue> part_of_explicit_cast{{$}}
// CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'const char *' lvalue Var {{.*}} 'const_char_ptr_value' 'const char *'{{$}}

// CXXFunctionalCastExpr
functional_cast_result *double_ptr = &floating;
// CHECK: TypedefDecl {{.*}} <{{.*}}, col:{{.*}}> col:{{.*}} referenced functional_cast_result 'typeof (double(int_value))':'double'{{$}}
// CHECK-NEXT: TypeOfExprType {{.*}} 'typeof (double(int_value))' sugar{{( imported)?}}{{$}}
// CHECK-NEXT: ParenExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'double'{{$}}
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} <col:{{.*}}, col:{{.*}}> 'double' functional cast to double <NoOp>{{$}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'double' <IntegralToFloating> part_of_explicit_cast{{$}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue> part_of_explicit_cast{{$}}
// CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue Var {{.*}} 'int_value' 'int'{{$}}

// CXXBoolLiteralExpr
bool_literal_result *bool_ptr = &boolean;
static_assert(true_value, "true_value is true");
static_assert(!false_value, "false_value is false");

// CXXNullPtrLiteralExpr
cxx_null_ptr_result null_ptr = nullptr;

// CXXTypeidExpr
typeid_result1 typeid_1 = 0;
typeid_result2 typeid_2 = 0;

// CharacterLiteral variants
static_assert(char_value == 97, "char_value is correct");
static_assert(wchar_t_value == 305, "wchar_t_value is correct");
static_assert(char16_t_value == 231, "char16_t_value is correct");
static_assert(char32_t_value == 8706, "char32_t_value is correct");
