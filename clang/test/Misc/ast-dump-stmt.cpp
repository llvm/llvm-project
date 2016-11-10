// RUN: %clang_cc1 -fcxx-exceptions -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

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
// CHECK-NEXT:      CompoundStmt
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
