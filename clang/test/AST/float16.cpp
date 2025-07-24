// Tests without serialization:
// RUN: %clang_cc1 -std=c++11 -ast-dump -triple aarch64-linux-gnu %s \
// RUN: | FileCheck %s --strict-whitespace
//
// RUN: %clang_cc1 -std=c++11 -ast-dump -triple aarch64-linux-gnu -fnative-half-type %s \
// RUN: | FileCheck %s --check-prefix=CHECK-NATIVE --strict-whitespace
//
// Tests with serialization:
// RUN: %clang_cc1 -std=c++11 -triple aarch64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -triple aarch64-linux-gnu -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s --strict-whitespace
//
// RUN: %clang_cc1 -std=c++11 -triple aarch64-linux-gnu -fnative-half-type -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -triple aarch64-linux-gnu -fnative-half-type -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s --check-prefix=CHECK-NATIVE --strict-whitespace

/*  Various contexts where type _Float16 can appear. */

/*  Namespace */
namespace {
  _Float16 f1n;
  _Float16 f2n = 33.f16;
  _Float16 arr1n[10];
  _Float16 arr2n[] = { 1.2, 3.0, 3.e4 };
  const volatile _Float16 func1n(const _Float16 &arg) {
    return arg + f2n + arr1n[4] - arr2n[1];
  }
}

// CHECK: |-NamespaceDecl {{.*}} 
// CHECK-NEXT: | |-VarDecl {{.*}} used f1n '_Float16'
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK-NEXT: | |-VarDecl {{.*}} used f2n '_Float16' cinit
// CHECK-NEXT: | | |-FloatingLiteral {{.*}} '_Float16' 3.300000e+01
// CHECK-NEXT: | | `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK-NEXT: | |-VarDecl {{.*}} used arr1n '_Float16[10]'
// CHECK-NEXT: | | `-typeDetails: ConstantArrayType {{.*}} '_Float16[10]' 10
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK-NEXT: | |-VarDecl {{.*}} used arr2n '_Float16[3]' cinit
// CHECK-NEXT: | | |-InitListExpr {{.*}} '_Float16[3]'
// CHECK-NEXT: | | | |-ImplicitCastExpr {{.*}} '_Float16' <FloatingCast>
// CHECK-NEXT: | | | | `-FloatingLiteral {{.*}} 'double' 1.200000e+00
// CHECK-NEXT: | | | |-ImplicitCastExpr {{.*}} '_Float16' <FloatingCast>
// CHECK-NEXT: | | | | `-FloatingLiteral {{.*}} 'double' 3.000000e+00
// CHECK-NEXT: | | | `-ImplicitCastExpr {{.*}} '_Float16' <FloatingCast>
// CHECK-NEXT: | | |   `-FloatingLiteral {{.*}} 'double' 3.000000e+04
// CHECK-NEXT: | | `-typeDetails: ConstantArrayType {{.*}} '_Float16[3]' 3
// CHECK-NEXT: | |   `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK-NEXT: | `-FunctionDecl {{.*}} line:27:27 used func1n 'const volatile _Float16 (const _Float16 &)'
// CHECK-NEXT: |   |-ParmVarDecl {{.*}} used arg 'const _Float16 &'
// CHECK-NEXT: |   | `-typeDetails: LValueReferenceType {{.*}} 'const _Float16 &'
// CHECK-NEXT: |   |   `-qualTypeDetail: QualType {{.*}} 'const _Float16' const
// CHECK-NEXT: |   |     `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK-NEXT: |   `-CompoundStmt {{.*}}
// CHECK-NEXT: |     `-ReturnStmt {{.*}} 
// CHECK-NEXT: |       `-BinaryOperator {{.*}} '_Float16' '-'
// CHECK-NEXT: |         |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |         | |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |         | | |-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |         | | | `-DeclRefExpr {{.*}} 'const _Float16' lvalue ParmVar {{.*}} 'arg' 'const _Float16 &'
// CHECK-NEXT: |         | | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |         | |   `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f2n' '_Float16'
// CHECK-NEXT: |         | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |         |   `-ArraySubscriptExpr {{.*}} '_Float16' lvalue
// CHECK-NEXT: |         |     |-ImplicitCastExpr {{.*}} '_Float16 *' <ArrayToPointerDecay>
// CHECK-NEXT: |         |     | `-DeclRefExpr {{.*}} '_Float16[10]' lvalue Var {{.*}} 'arr1n' '_Float16[10]'
// CHECK-NEXT: |         |     `-IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: |         `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |           `-ArraySubscriptExpr {{.*}} '_Float16' lvalue
// CHECK-NEXT: |             |-ImplicitCastExpr {{.*}} '_Float16 *' <ArrayToPointerDecay>
// CHECK-NEXT: |             | `-DeclRefExpr {{.*}} '_Float16[3]' lvalue Var {{.*}} 'arr2n' '_Float16[3]'
// CHECK-NEXT: |             `-IntegerLiteral {{.*}} 'int' 1

/* File */
_Float16 f1f;
// CHECK: |-VarDecl {{.*}} f1f '_Float16'
// CHECK: | `-typeDetails: BuiltinType {{.*}} '_Float16'

_Float16 f2f = 32.4;
// CHECK: |-VarDecl {{.*}} f2f '_Float16' cinit
// CHECK-NEXT: | |-ImplicitCastExpr {{.*}} <col:16> '_Float16' <FloatingCast>
// CHECK-NEXT: | | `-FloatingLiteral {{.*}} <col:16> 'double' 3.240000e+01
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} '_Float16'

_Float16 arr1f[10];
//CHECK: |-VarDecl {{.*}} arr1f '_Float16[10]'

_Float16 arr2f[] = { -1.2, -3.0, -3.e4 };
// CHECK: |-VarDecl {{.*}} col:10 arr2f '_Float16[3]' cinit
// CHECK: | |-InitListExpr {{.*}} <col:20, col:40> '_Float16[3]'
// CHECK: | | |-ImplicitCastExpr {{.*}} <col:22, col:23> '_Float16' <FloatingCast>
// CHECK: | | | `-UnaryOperator {{.*}} <col:22, col:23> 'double' prefix '-'
// CHECK: | | |   `-FloatingLiteral {{.*}} <col:23> 'double' 1.200000e+00
// CHECK: | | |-ImplicitCastExpr {{.*}} <col:28, col:29> '_Float16' <FloatingCast>
// CHECK: | | | `-UnaryOperator {{.*}} <col:28, col:29> 'double' prefix '-'
// CHECK: | | |   `-FloatingLiteral {{.*}} <col:29> 'double' 3.000000e+00
// CHECK: | | `-ImplicitCastExpr {{.*}} <col:34, col:35> '_Float16' <FloatingCast>
// CHECK: | |   `-UnaryOperator {{.*}} <col:34, col:35> 'double' prefix '-'
// CHECK: | |     `-FloatingLiteral {{.*}} <col:35> 'double' 3.000000e+04
// CHECK: | `-typeDetails: ConstantArrayType {{.*}} '_Float16[3]' 3
// CHECK: |   `-typeDetails: BuiltinType {{.*}} '_Float16'

_Float16 func1f(_Float16 arg);

//CHECK: |-FunctionDecl {{.*}} func1f '_Float16 (_Float16)'
//CHECK: | `-ParmVarDecl {{.*}} arg '_Float16'


// Mixing __fp16 and Float16 types:
// The _Float16 type is first converted to __fp16 type and then the operation
// is completed as if both operands were of __fp16 type.

__fp16 B = -0.1;
auto C = -1.0f16 + B;

// When we do *not* have native half types, we expect __fp16 to be promoted to
// float, and consequently also _Float16 promotions to float:
// CHECK: |-VarDecl {{.*}} used B '__fp16' cinit
// CHECK-NEXT: | |-ImplicitCastExpr {{.*}} '__fp16' <FloatingCast>
// CHECK-NEXT: | | `-UnaryOperator {{.*}} 'double' prefix '-'
// CHECK-NEXT: | |   `-FloatingLiteral {{.*}} 'double' 1.000000e-01
// CHECK-NEXT: | `-typeDetails: BuiltinType {{.*}} '__fp16'
// CHECK-NEXT: |-VarDecl {{.*}} col:6 C 'float' cinit
// CHECK-NEXT: | |-BinaryOperator {{.*}} 'float' '+'
// CHECK-NEXT: | | |-ImplicitCastExpr {{.*}} 'float' <FloatingCast>
// CHECK-NEXT: | | | `-UnaryOperator {{.*}} '_Float16' prefix '-'
// CHECK-NEXT: | | |   `-FloatingLiteral {{.*}} '_Float16' 1.000000e+00
// CHECK-NEXT: | | `-ImplicitCastExpr {{.*}} 'float' <FloatingCast>
// CHECK-NEXT: | |   `-ImplicitCastExpr {{.*}} '__fp16' <LValueToRValue>
// CHECK-NEXT: | |     `-DeclRefExpr {{.*}} '__fp16' lvalue Var {{.*}} 'B' '__fp16'
// CHECK-NEXT: | `-typeDetails: AutoType {{.*}} 'float' sugar
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} 'float'

// When we do have native half types, we expect to see promotions to fp16:

// CHECK-NATIVE: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// CHECK-NATIVE: |-VarDecl {{.*}} used B '__fp16' cinit
// CHECK-NATIVE: | |-ImplicitCastExpr {{.*}} '__fp16' <FloatingCast>
// CHECK-NATIVE: | | `-UnaryOperator {{.*}} 'double' prefix '-'
// CHECK-NATIVE: | |   `-FloatingLiteral {{.*}} 'double' 1.000000e-01
// CHECK-NATIVE: | `-typeDetails: BuiltinType {{.*}} '__fp16'
// CHECK-NATIVE: |-VarDecl {{.*}} C '__fp16' cinit
// CHECK-NATIVE: | |-BinaryOperator {{.*}} '__fp16' '+'
// CHECK-NATIVE: | | |-ImplicitCastExpr {{.*}} '__fp16' <FloatingCast>
// CHECK-NATIVE: | | | `-UnaryOperator {{.*}} '_Float16' prefix '-'
// CHECK-NATIVE: | | |   `-FloatingLiteral {{.*}} '_Float16' 1.000000e+00
// CHECK-NATIVE: | | `-ImplicitCastExpr {{.*}} '__fp16' <LValueToRValue>
// CHECK-NATIVE: | |   `-DeclRefExpr {{.*}} '__fp16' lvalue Var {{.*}} 'B' '__fp16'
// CHECK-NATIVE: | `-typeDetails: AutoType {{.*}} '__fp16' sugar
// CHECK-NATIVE: |   `-typeDetails: BuiltinType {{.*}} '__fp16'

/* Class */

class C1 {
  _Float16 f1c;
  static const _Float16 f2c;
  volatile _Float16 f3c;
public:
  C1(_Float16 arg) : f1c(arg), f3c(arg) { }
  _Float16 func1c(_Float16 arg ) {
    return f1c + arg;
  }
  static _Float16 func2c(_Float16 arg) {
    return arg * C1::f2c;
  }
};

//CHECK:      |-CXXRecordDecl {{.*}} referenced class C1 definition
//CHECK:      | |-CXXRecordDecl {{.*}} implicit referenced class C1
//CHECK-NEXT: | |-FieldDecl {{.*}} referenced f1c '_Float16'
//CHECK-NEXT: | |-VarDecl {{.*}} used f2c 'const _Float16' static
//CHECK: | |-FieldDecl {{.*}} f3c 'volatile _Float16'
//CHECK-NEXT: | |-AccessSpecDecl
//CHECK-NEXT: | |-CXXConstructorDecl {{.*}} used C1 'void (_Float16)
//CHECK-NEXT: | | |-ParmVarDecl {{.*}} used arg '_Float16'
//CHECK: | | |-CXXCtorInitializer Field {{.*}} 'f1c' '_Float16'
//CHECK-NEXT: | | | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: | | |   `-DeclRefExpr {{.*}} '_Float16' lvalue ParmVar 0x{{.*}} 'arg' '_Float16'
//CHECK-NEXT: | | |-CXXCtorInitializer Field {{.*}} 'f3c' 'volatile _Float16'
//CHECK-NEXT: | | | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: | | |   `-DeclRefExpr {{.*}} '_Float16' lvalue ParmVar 0x{{.*}} 'arg' '_Float16'
//CHECK-NEXT: | | `-CompoundStmt
//CHECK-NEXT: | |-CXXMethodDecl {{.*}} used func1c '_Float16 (_Float16)
//CHECK-NEXT: | | |-ParmVarDecl {{.*}} used arg '_Float16'
//CHECK: | | `-CompoundStmt
//CHECK-NEXT: | |   `-ReturnStmt
//CHECK-NEXT: | |     `-BinaryOperator {{.*}} '_Float16' '+'
//CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: | |       | `-MemberExpr {{.*}} '_Float16' lvalue ->f1c 0x{{.*}}
//CHECK-NEXT: | |       |   `-CXXThisExpr {{.*}} 'C1 *' implicit this
//CHECK-NEXT: | |       `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: | |         `-DeclRefExpr {{.*}} '_Float16' lvalue ParmVar 0x{{.*}} 'arg' '_Float16'
//CHECK-NEXT: | |-CXXMethodDecl {{.*}} used func2c '_Float16 (_Float16)' static
//CHECK-NEXT: | | |-ParmVarDecl {{.*}} used arg '_Float16'
//CHECK: | | `-CompoundStmt
//CHECK-NEXT: | |   `-ReturnStmt
//CHECK-NEXT: | |     `-BinaryOperator {{.*}} '_Float16' '*'
//CHECK-NEXT: | |       |-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: | |       | `-DeclRefExpr {{.*}} '_Float16' lvalue ParmVar 0x{{.*}} 'arg' '_Float16'
//CHECK-NEXT: | |       `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: | |         `-DeclRefExpr {{.*}} 'const _Float16' lvalue Var 0x{{.*}} 'f2c' 'const _Float16'


/*  Template */

template <class C> C func1t(C arg) {
  return arg * 2.f16;
}

//CHECK:      |-FunctionTemplateDecl {{.*}} func1t
//CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} C
//CHECK-NEXT: | |-FunctionDecl {{.*}} func1t 'C (C)'
//CHECK-NEXT: | | |-ParmVarDecl {{.*}} referenced arg 'C'
//CHECK: | | `-CompoundStmt
//CHECK-NEXT: | |   `-ReturnStmt
//CHECK-NEXT: | |     `-BinaryOperator {{.*}} '<dependent type>' '*'
//CHECK-NEXT: | |       |-DeclRefExpr {{.*}} 'C' lvalue ParmVar {{.*}} 'arg' 'C'
//CHECK-NEXT: | |       `-FloatingLiteral {{.*}} '_Float16' 2.000000e+00
//CHECK-NEXT: | `-FunctionDecl {{.*}} used func1t '_Float16 (_Float16)'
//CHECK-NEXT: |   |-TemplateArgument type '_Float16'
//CHECK:      |   |-ParmVarDecl {{.*}} used arg '_Float16'
//CHECK: |   `-CompoundStmt
//CHECK-NEXT: |     `-ReturnStmt
//CHECK-NEXT: |       `-BinaryOperator {{.*}} '_Float16' '*'
//CHECK-NEXT: |         |-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
//CHECK-NEXT: |         | `-DeclRefExpr {{.*}} '_Float16' lvalue ParmVar {{.*}} 'arg' '_Float16'
//CHECK-NEXT: |         `-FloatingLiteral {{.*}} '_Float16' 2.000000e+00


template <class C> struct S1 {
  C mem1;
};

//CHECK:      |-ClassTemplateDecl {{.*}} S1
//CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} referenced class depth 0 index 0 C
//CHECK-NEXT: | |-CXXRecordDecl {{.*}} struct S1 definition
//CHECK:      | | |-CXXRecordDecl {{.*}} implicit struct S1
//CHECK-NEXT: | | `-FieldDecl {{.*}} mem1 'C'
//CHECK-NEXT: | `-ClassTemplateSpecialization {{.*}} 'S1'

template <> struct S1<_Float16> {
  _Float16 mem2;
};


/* Local */

extern int printf (const char *__restrict __format, ...);

int main(void) {
  _Float16 f1l = 1e3f16;
//CHECK:       | `-VarDecl {{.*}} used f1l '_Float16' cinit
//CHECK-NEXT:  |   |-FloatingLiteral {{.*}} '_Float16' 1.000000e+03

  _Float16 f2l = -0.f16;
// CHECK: | `-VarDecl {{.*}} used f2l '_Float16' cinit
// CHECK-NEXT: |   |-UnaryOperator {{.*}} '_Float16' prefix '-'
// CHECK-NEXT: |   | `-FloatingLiteral {{.*}} '_Float16' 0.000000e+00
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} '_Float16'

  _Float16 f3l = 1.000976562;
// CHECK: | `-VarDecl {{.*}} used f3l '_Float16' cinit
// CHECK-NEXT: |   |-ImplicitCastExpr {{.*}} '_Float16' <FloatingCast>
// CHECK-NEXT: |   | `-FloatingLiteral {{.*}} 'double' 1.000977e+00
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} '_Float16'

  C1 c1(f1l);
// CHECK: | `-VarDecl {{.*}} used c1 'C1' callinit
// CHECK-NEXT: |   |-CXXConstructExpr {{.*}} 'C1' 'void (_Float16)'
// CHECK-NEXT: |   | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f1l' '_Float16'
// CHECK-NEXT: |   `-typeDetails: ElaboratedType {{.*}} 'C1' sugar
// CHECK-NEXT: |     `-typeDetails: RecordType {{.*}} 'C1'
// CHECK-NEXT: |       `-CXXRecord {{.*}} 'C1'

  S1<_Float16> s1 = { 132.f16 };
// CHECK: | `-VarDecl {{.*}} used s1 'S1<_Float16>' cinit
// CHECK-NEXT: |   |-InitListExpr {{.*}} 'S1<_Float16>'
// CHECK-NEXT: |   | `-FloatingLiteral {{.*}} '_Float16' 1.320000e+02
// CHECK-NEXT: |   `-typeDetails: ElaboratedType {{.*}} 'S1<_Float16>' sugar
// CHECK-NEXT: |     `-typeDetails: TemplateSpecializationType {{.*}} 'S1<_Float16>' sugar
// CHECK-NEXT: |       |-name: 'S1' qualified
// CHECK-NEXT: |       | `-ClassTemplateDecl {{.*}} S1
// CHECK-NEXT: |       |-TemplateArgument type '_Float16'
// CHECK-NEXT: |       | `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK-NEXT: |       `-typeDetails: RecordType {{.*}} 'S1<_Float16>'
// CHECK-NEXT: |         `-ClassTemplateSpecialization {{.*}} 'S1'

  _Float16 f4l = func1n(f1l)  + func1f(f2l) + c1.func1c(f3l) + c1.func2c(f1l) +
    func1t(f1l) + s1.mem2 - f1n + f2n;
// CHECK: | `-VarDecl {{.*}} used f4l '_Float16' cinit
// CHECK-NEXT: |   |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |   | |-BinaryOperator {{.*}} '_Float16' '-'
// CHECK-NEXT: |   | | |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |   | | | |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |   | | | | |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |   | | | | | |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |   | | | | | | |-BinaryOperator {{.*}} '_Float16' '+'
// CHECK-NEXT: |   | | | | | | | |-CallExpr {{.*}} '_Float16'
// CHECK-NEXT: |   | | | | | | | | |-ImplicitCastExpr {{.*}} 'const volatile _Float16 (*)(const _Float16 &)' <FunctionToPointerDecay>
// CHECK-NEXT: |   | | | | | | | | | `-DeclRefExpr {{.*}} 'const volatile _Float16 (const _Float16 &)' lvalue Function {{.*}} 'func1n' 'const volatile _Float16 (const _Float16 &)'
// CHECK-NEXT: |   | | | | | | | | `-ImplicitCastExpr {{.*}} 'const _Float16' lvalue <NoOp>
// CHECK-NEXT: |   | | | | | | | |   `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f1l' '_Float16'
// CHECK-NEXT: |   | | | | | | | `-CallExpr {{.*}} '_Float16'
// CHECK-NEXT: |   | | | | | | |   |-ImplicitCastExpr {{.*}} '_Float16 (*)(_Float16)' <FunctionToPointerDecay>
// CHECK-NEXT: |   | | | | | | |   | `-DeclRefExpr {{.*}} '_Float16 (_Float16)' lvalue Function {{.*}} 'func1f' '_Float16 (_Float16)'
// CHECK-NEXT: |   | | | | | | |   `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   | | | | | | |     `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f2l' '_Float16'
// CHECK-NEXT: |   | | | | | | `-CXXMemberCallExpr {{.*}} '_Float16'
// CHECK-NEXT: |   | | | | | |   |-MemberExpr {{.*}} '<bound member function type>' .func1c {{.*}}
// CHECK-NEXT: |   | | | | | |   | `-DeclRefExpr {{.*}} 'C1' lvalue Var {{.*}} 'c1' 'C1'
// CHECK-NEXT: |   | | | | | |   `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   | | | | | |     `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f3l' '_Float16'
// CHECK-NEXT: |   | | | | | `-CallExpr {{.*}} '_Float16'
// CHECK-NEXT: |   | | | | |   |-ImplicitCastExpr {{.*}} '_Float16 (*)(_Float16)' <FunctionToPointerDecay>
// CHECK-NEXT: |   | | | | |   | `-MemberExpr {{.*}} '_Float16 (_Float16)' lvalue .func2c {{.*}}
// CHECK-NEXT: |   | | | | |   |   `-DeclRefExpr {{.*}} 'C1' lvalue Var {{.*}} 'c1' 'C1'
// CHECK-NEXT: |   | | | | |   `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   | | | | |     `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f1l' '_Float16'
// CHECK-NEXT: |   | | | | `-CallExpr {{.*}} '_Float16'
// CHECK-NEXT: |   | | | |   |-ImplicitCastExpr {{.*}} '_Float16 (*)(_Float16)' <FunctionToPointerDecay>
// CHECK-NEXT: |   | | | |   | `-DeclRefExpr {{.*}} '_Float16 (_Float16)' lvalue Function {{.*}} 'func1t' '_Float16 (_Float16)' (FunctionTemplate {{.*}} 'func1t')
// CHECK-NEXT: |   | | | |   `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   | | | |     `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f1l' '_Float16'
// CHECK-NEXT: |   | | | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   | | |   `-MemberExpr {{.*}} '_Float16' lvalue .mem2 {{.*}}
// CHECK-NEXT: |   | | |     `-DeclRefExpr {{.*}} 'S1<_Float16>' lvalue Var {{.*}} 's1' 'S1<_Float16>'
// CHECK-NEXT: |   | | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   | |   `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f1n' '_Float16'
// CHECK-NEXT: |   | `-ImplicitCastExpr {{.*}} '_Float16' <LValueToRValue>
// CHECK-NEXT: |   |   `-DeclRefExpr {{.*}} '_Float16' lvalue Var {{.*}} 'f2n' '_Float16'
// CHECK-NEXT: |   `-typeDetails: BuiltinType {{.*}} '_Float16'

  auto f5l = -1.f16, *f6l = &f2l, f7l = func1t(f3l);

// CHECK:  | |-VarDecl {{.*}} <col:3, col:15> col:8 f5l '_Float16' cinit
// CHECK:  | | |-UnaryOperator {{.*}} <col:14, col:15> '_Float16' prefix '-'
// CHECK:  | | | `-FloatingLiteral {{.*}} <col:15> '_Float16' 1.000000e+00
// CHECK:  | | `-typeDetails: AutoType {{.*}} '_Float16' sugar
// CHECK:  | |   `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK:  | |-VarDecl {{.*}} <col:3, col:30> col:23 f6l '_Float16 *' cinit
// CHECK:  | | |-UnaryOperator {{.*}} <col:29, col:30> '_Float16 *' prefix '&' cannot overflow
// CHECK:  | | | `-DeclRefExpr {{.*}} <col:30> '_Float16' lvalue Var {{.*}} 'f2l' '_Float16'
// CHECK:  | | `-typeDetails: PointerType {{.*}} '_Float16 *'
// CHECK:  | |   `-typeDetails: AutoType {{.*}} '_Float16' sugar
// CHECK:  | |     `-typeDetails: BuiltinType {{.*}} '_Float16'
// CHECK:  | `-VarDecl {{.*}} <col:3, col:51> col:35 f7l '_Float16' cinit
// CHECK:  |   |-CallExpr {{.*}} <col:41, col:51> '_Float16'
// CHECK:  |   | |-ImplicitCastExpr {{.*}} <col:41> '_Float16 (*)(_Float16)' <FunctionToPointerDecay>
// CHECK:  |   | | `-DeclRefExpr {{.*}} <col:41> '_Float16 (_Float16)' lvalue Function {{.*}} 'func1t' '_Float16 (_Float16)' (FunctionTemplate {{.*}} 'func1t')
// CHECK:  |   | `-ImplicitCastExpr {{.*}} <col:48> '_Float16' <LValueToRValue>
// CHECK:  |   |   `-DeclRefExpr {{.*}} <col:48> '_Float16' lvalue Var {{.*}} 'f3l' '_Float16'
// CHECK:  |   `-typeDetails: AutoType {{.*}} '_Float16' sugar
// CHECK:  |     `-typeDetails: SubstTemplateTypeParmType {{.*}} '_Float16' sugar class depth 0 index 0 C
// CHECK:  |       |-FunctionTemplate {{.*}} 'func1t'
// CHECK:  |       `-typeDetails: BuiltinType {{.*}} '_Float16'

  _Float16 f8l = f4l++;
// CHECK: | `-VarDecl {{.*}} <col:3, col:21> col:12 f8l '_Float16' cinit
// CHECK: |   |-UnaryOperator {{.*}} <col:18, col:21> '_Float16' postfix '++'
// CHECK: |   | `-DeclRefExpr {{.*}} <col:18> '_Float16' lvalue Var {{.*}} 'f4l' '_Float16'
// CHECK: |   `-typeDetails: BuiltinType {{.*}} '_Float16'


  _Float16 arr1l[] = { -1.f16, -0.f16, -11.f16 };
// CHECK: | `-VarDecl {{.*}} <col:3, col:48> col:12 arr1l '_Float16[3]' cinit
// CHECK: |   |-InitListExpr {{.*}} <col:22, col:48> '_Float16[3]'
// CHECK: |   | |-UnaryOperator {{.*}} <col:24, col:25> '_Float16' prefix '-'
// CHECK: |   | | `-FloatingLiteral {{.*}} <col:25> '_Float16' 1.000000e+00
// CHECK: |   | |-UnaryOperator {{.*}} <col:32, col:33> '_Float16' prefix '-'
// CHECK: |   | | `-FloatingLiteral {{.*}} <col:33> '_Float16' 0.000000e+00
// CHECK: |   | `-UnaryOperator {{.*}} <col:40, col:41> '_Float16' prefix '-'
// CHECK: |   |   `-FloatingLiteral {{.*}} <col:41> '_Float16' 1.100000e+01
// CHECK: |   `-typeDetails: ConstantArrayType {{.*}} '_Float16[3]' 3
// CHECK: |     `-typeDetails: BuiltinType {{.*}} '_Float16'

  float cvtf = f2n;
// CHECK: | `-VarDecl {{.*}} <col:3, col:16> col:9 cvtf 'float' cinit
// CHECK: |   |-ImplicitCastExpr {{.*}} <col:16> 'float' <FloatingCast>
// CHECK: |   | `-ImplicitCastExpr {{.*}} <col:16> '_Float16' <LValueToRValue>
// CHECK: |   |   `-DeclRefExpr {{.*}} <col:16> '_Float16' lvalue Var {{.*}} 'f2n' '_Float16'
// CHECK: |   `-typeDetails: BuiltinType {{.*}} 'float'

  double cvtd = f2n;
// CHECK: | `-VarDecl {{.*}} <col:3, col:17> col:10 cvtd 'double' cinit
// CHECK: |   |-ImplicitCastExpr {{.*}} <col:17> 'double' <FloatingCast>
// CHECK: |   | `-ImplicitCastExpr {{.*}} <col:17> '_Float16' <LValueToRValue>
// CHECK: |   |   `-DeclRefExpr {{.*}} <col:17> '_Float16' lvalue Var {{.*}} 'f2n' '_Float16'
// CHECK: |   `-typeDetails: BuiltinType {{.*}} 'double'

  long double cvtld = f2n;
// CHECK: | `-VarDecl {{.*}} <col:3, col:23> col:15 cvtld 'long double' cinit
// CHECK: |   |-ImplicitCastExpr {{.*}} <col:23> 'long double' <FloatingCast>
// CHECK: |   | `-ImplicitCastExpr {{.*}} <col:23> '_Float16' <LValueToRValue>
// CHECK: |   |   `-DeclRefExpr {{.*}} <col:23> '_Float16' lvalue Var {{.*}} 'f2n' '_Float16'
// CHECK: |   `-typeDetails: BuiltinType {{.*}} 'long double'

  _Float16 f2h = 42.0f;
// CHECK: | `-VarDecl {{.*}} <col:3, col:18> col:12 f2h '_Float16' cinit
// CHECK: |   |-ImplicitCastExpr {{.*}} <col:18> '_Float16' <FloatingCast>
// CHECK: |   | `-FloatingLiteral {{.*}} <col:18> 'float' 4.200000e+01
// CHECK: |   `-typeDetails: BuiltinType {{.*}} '_Float16'

  _Float16 d2h = 42.0;
// CHECK: | `-VarDecl {{.*}} <col:3, col:18> col:12 d2h '_Float16' cinit
// CHECK: |   |-ImplicitCastExpr {{.*}} <col:18> '_Float16' <FloatingCast>
// CHECK: |   | `-FloatingLiteral {{.*}} <col:18> 'double' 4.200000e+01
// CHECK: |   `-typeDetails: BuiltinType {{.*}} '_Float16'

  _Float16 ld2h = 42.0l;
// CHECK: `-VarDecl {{.*}} <col:3, col:19> col:12 ld2h '_Float16' cinit
// CHECK:   |-ImplicitCastExpr {{.*}} <col:19> '_Float16' <FloatingCast>
// CHECK:   | `-FloatingLiteral {{.*}} <col:19> 'long double' 4.200000e+01
// CHECK:   `-typeDetails: BuiltinType {{.*}} '_Float16'
}
