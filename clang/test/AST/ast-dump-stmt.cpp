// Test without serialization:
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -fcxx-exceptions -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++23 -triple x86_64-linux-gnu -fcxx-exceptions -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck -strict-whitespace %s

namespace n {
void function() {}
int Variable;
}
using n::function;
using n::Variable;
void TestFunction() {
  void (*f)() = &function;
// CHECK:       DeclRefExpr{{.*}} (UsingShadow{{.*}}function
  Variable = 4;
// CHECK:       DeclRefExpr{{.*}} (UsingShadow{{.*}}Variable
}

// CHECK: FunctionDecl {{.*}} TestCatch1
void TestCatch1() {
// CHECK:       CXXTryStmt
// CHECK-NEXT:    CompoundStmt
  try {
  }
// CHECK-NEXT:    CXXCatchStmt
// CHECK-NEXT:      VarDecl {{.*}} x
// CHECK:      CompoundStmt
  catch (int x) {
  }
}

// CHECK: FunctionDecl {{.*}} TestCatch2
void TestCatch2() {
// CHECK:       CXXTryStmt
// CHECK-NEXT:    CompoundStmt
  try {
  }
// CHECK-NEXT:    CXXCatchStmt
// CHECK-NEXT:      NULL
// CHECK-NEXT:      CompoundStmt
  catch (...) {
  }
}

void TestAllocationExprs() {
  int *p;
  p = new int;
  delete p;
  p = new int[2];
  delete[] p;
  p = ::new int;
  ::delete p;
}
// CHECK: FunctionDecl {{.*}} TestAllocationExprs
// CHECK: CXXNewExpr {{.*}} 'int *' Function {{.*}} 'operator new'
// CHECK: CXXDeleteExpr {{.*}} 'void' Function {{.*}} 'operator delete'
// CHECK: CXXNewExpr {{.*}} 'int *' array Function {{.*}} 'operator new[]'
// CHECK: CXXDeleteExpr {{.*}} 'void' array Function {{.*}} 'operator delete[]'
// CHECK: CXXNewExpr {{.*}} 'int *' global Function {{.*}} 'operator new'
// CHECK: CXXDeleteExpr {{.*}} 'void' global Function {{.*}} 'operator delete'

// Don't crash on dependent exprs that haven't been resolved yet.
template <typename T>
void TestDependentAllocationExpr() {
  T *p = new T;
  delete p;
}
// CHECK: FunctionTemplateDecl {{.*}} TestDependentAllocationExpr
// CHECK: CXXNewExpr {{.*'T \*'$}}
// CHECK: CXXDeleteExpr {{.*'void'$}}

template <typename T>
class DependentScopeMemberExprWrapper {
  T member;
};

template <typename T>
void TestDependentScopeMemberExpr() {
  DependentScopeMemberExprWrapper<T> obj;
  obj.member = T();
  (&obj)->member = T();
}
// CHECK: FunctionTemplateDecl {{.*}} TestDependentScopeMemberExpr
// CHECK: CXXDependentScopeMemberExpr {{.*}} lvalue .member
// CHECK: CXXDependentScopeMemberExpr {{.*}} lvalue ->member

union U {
  int i;
  long l;
};

void TestUnionInitList()
{
  U us[3] = {1};
// CHECK: VarDecl {{.+}} <col:3, col:15> col:5 us 'U[3]' cinit
// CHECK-NEXT: |-InitListExpr {{.+}} <col:13, col:15> 'U[3]'
// CHECK-NEXT:   |-array_filler: InitListExpr {{.+}} <col:15> 'U' field Field {{.+}} 'i' 'int'
// CHECK-NEXT:   `-InitListExpr {{.+}} <col:14> 'U' field Field {{.+}} 'i' 'int'
// CHECK-NEXT:     `-IntegerLiteral {{.+}} <col:14> 'int' 1
}

void TestSwitch(int i) {
  switch (int a; i)
    ;
  // CHECK: SwitchStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:5> has_init
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:11, col:15> col:15 a 'int'
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:18> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
  // CHECK: NullStmt
}

void TestIf(bool b) {
  if (int i = 12; b)
    ;
  // CHECK: IfStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:5> has_init
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:7, col:15> col:11 i 'int' cinit
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:15> 'int' 12
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:19> 'bool' lvalue ParmVar 0x{{[^ ]*}} 'b' 'bool'
  // CHECK: NullStmt

  if constexpr (sizeof(b) == 1)
    ;
  // CHECK: IfStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:5>
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <line:[[@LINE-3]]:17, col:30> 'bool'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: BinaryOperator
  // CHECK-NEXT: UnaryExprOrTypeTraitExpr
  // CHECK-NEXT: ParenExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: NullStmt

  if constexpr (sizeof(b) == 1)
    ;
  else
    ;
  // CHECK: IfStmt 0x{{[^ ]*}} <line:[[@LINE-4]]:3, line:[[@LINE-1]]:5> has_else
  // CHECK-NEXT: ConstantExpr 0x{{[^ ]*}} <line:[[@LINE-5]]:17, col:30> 'bool'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: BinaryOperator
  // CHECK-NEXT: UnaryExprOrTypeTraitExpr
  // CHECK-NEXT: ParenExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: NullStmt
  // CHECK-NEXT: NullStmt

  if consteval {}
  // CHECK: IfStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:17> consteval
  // CHECK-NEXT: CompoundStmt

  if ! consteval {}
  else {}
  // CHECK: IfStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:9> has_else !consteval
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: CompoundStmt
}

struct Container {
  int *begin() const;
  int *end() const;
};

void TestIteration() {
  for (int i = 0; int j = i; ++i)
    ;
  // CHECK: ForStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:5>
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:8, col:16> col:12 used i 'int' cinit
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:16> 'int' 0
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:19, col:27> col:23 used j 'int' cinit
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:27> 'int' lvalue Var 0x{{[^ ]*}} 'i' 'int'
  // CHECK: ImplicitCastExpr 0x{{[^ ]*}} <col:23> 'bool' <IntegralToBoolean>
  // CHECK: ImplicitCastExpr 0x{{[^ ]*}} <col:23> 'int' <LValueToRValue>
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:23> 'int' lvalue Var 0x{{[^ ]*}} 'j' 'int'
  // CHECK: UnaryOperator 0x{{[^ ]*}} <col:30, col:32> 'int' lvalue prefix '++'
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:32> 'int' lvalue Var 0x{{[^ ]*}} 'i' 'int'
  // CHECK: NullStmt

  int vals[10];
  for (int v : vals)
    ;
  // CHECK: CXXForRangeStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:5>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:16> col:16 implicit used __range1 'int (&)[10]' cinit
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:16> 'int[10]' lvalue Var 0x{{[^ ]*}} 'vals' 'int[10]'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:14> col:14 implicit used __begin1 'int *' cinit
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int[10]' lvalue Var 0x{{[^ ]*}} '__range1' 'int (&)[10]'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:14, col:16> col:14 implicit used __end1 'int *' cinit
  // CHECK: BinaryOperator 0x{{[^ ]*}} <col:14, col:16> 'int *' '+'
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int[10]' lvalue Var 0x{{[^ ]*}} '__range1' 'int (&)[10]'
  // CHECK: IntegerLiteral 0x{{[^ ]*}} <col:16> 'long' 10
  // CHECK: BinaryOperator 0x{{[^ ]*}} <col:14> 'bool' '!='
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__end1' 'int *'
  // CHECK: UnaryOperator 0x{{[^ ]*}} <col:14> 'int *' lvalue prefix '++'
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:8, col:14> col:12 v 'int' cinit
  // CHECK: ImplicitCastExpr
  // CHECK: UnaryOperator 0x{{[^ ]*}} <col:14> 'int' lvalue prefix '*' cannot overflow
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: NullStmt

  Container C;
  for (int v : C)
    ;
  // CHECK: CXXForRangeStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:5>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:16> col:16 implicit used __range1 'Container &' cinit
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:16> 'Container' lvalue Var 0x{{[^ ]*}} 'C' 'Container'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:14> col:14 implicit used __begin1 'int *' cinit
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <col:14> 'int *'
  // CHECK: MemberExpr 0x{{[^ ]*}} <col:14> '<bound member function type>' .begin 0x{{[^ ]*}}
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'Container' lvalue Var 0x{{[^ ]*}} '__range1' 'Container &'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:14> col:14 implicit used __end1 'int *' cinit
  // CHECK: CXXMemberCallExpr 0x{{[^ ]*}} <col:14> 'int *'
  // CHECK: MemberExpr 0x{{[^ ]*}} <col:14> '<bound member function type>' .end 0x{{[^ ]*}}
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'Container' lvalue Var 0x{{[^ ]*}} '__range1' 'Container &'
  // CHECK: BinaryOperator 0x{{[^ ]*}} <col:14> 'bool' '!='
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__end1' 'int *'
  // CHECK: UnaryOperator 0x{{[^ ]*}} <col:14> 'int *' lvalue prefix '++'
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:8, col:14> col:12 v 'int' cinit
  // CHECK: ImplicitCastExpr
  // CHECK: UnaryOperator 0x{{[^ ]*}} <col:14> 'int' lvalue prefix '*' cannot overflow
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:14> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: NullStmt

  for (int a; int v : vals)
    ;
  // CHECK: CXXForRangeStmt 0x{{[^ ]*}} <line:[[@LINE-2]]:3, line:[[@LINE-1]]:5>
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:8, col:12> col:12 a 'int'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:23> col:23 implicit used __range1 'int (&)[10]' cinit
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:23> 'int[10]' lvalue Var 0x{{[^ ]*}} 'vals' 'int[10]'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:21> col:21 implicit used __begin1 'int *' cinit
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:21> 'int[10]' lvalue Var 0x{{[^ ]*}} '__range1' 'int (&)[10]'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:21, col:23> col:21 implicit used __end1 'int *' cinit
  // CHECK: BinaryOperator 0x{{[^ ]*}} <col:21, col:23> 'int *' '+'
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:21> 'int[10]' lvalue Var 0x{{[^ ]*}} '__range1' 'int (&)[10]'
  // CHECK: IntegerLiteral 0x{{[^ ]*}} <col:23> 'long' 10
  // CHECK: BinaryOperator 0x{{[^ ]*}} <col:21> 'bool' '!='
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:21> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:21> 'int *' lvalue Var 0x{{[^ ]*}} '__end1' 'int *'
  // CHECK: UnaryOperator 0x{{[^ ]*}} <col:21> 'int *' lvalue prefix '++'
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:21> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: DeclStmt
  // CHECK: VarDecl 0x{{[^ ]*}} <col:15, col:21> col:19 v 'int' cinit
  // CHECK: ImplicitCastExpr
  // CHECK: UnaryOperator 0x{{[^ ]*}} <col:21> 'int' lvalue prefix '*' cannot overflow
  // CHECK: ImplicitCastExpr
  // CHECK: DeclRefExpr 0x{{[^ ]*}} <col:21> 'int *' lvalue Var 0x{{[^ ]*}} '__begin1' 'int *'
  // CHECK: NullStmt
}
