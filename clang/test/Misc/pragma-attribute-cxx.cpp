// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -fcxx-exceptions %s
// RUN: %clang_cc1 -ast-dump -ast-dump-filter test -std=c++11 -fcxx-exceptions %s | FileCheck %s
// expected-no-diagnostics

class testClass1 {
};
// CHECK-LABEL: CXXRecordDecl{{.*}} testClass1
// CHECK-NOT: AnnotateAttr

#pragma clang attribute push (__attribute__((annotate("test"))), apply_to=any(record, field, variable, function, namespace, type_alias))

class testClass2 {
  void testMethod1(int param);

  testClass2();

  testClass2 *operator -> ();
};
// CHECK-LABEL: CXXRecordDecl{{.*}} testClass2
// CHECK: AnnotateAttr{{.*}} "test"
// CHECK: CXXMethodDecl{{.*}} testMethod1
// CHECK-NEXT: ParmVarDecl{{.*}} param
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: CXXConstructorDecl
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: CXXMethodDecl{{.*}} operator->
// CHECK-NEXT: AnnotateAttr{{.*}} "test"

#pragma clang attribute push (__attribute__((annotate("method"))), apply_to=any(record, field, variable, function, namespace, type_alias))

void testClass2::testMethod1(int param) {

#pragma clang attribute pop
}
// CHECK-LABEL: CXXMethodDecl{{.*}}prev{{.*}} testMethod1
// CHECK-NEXT: ParmVarDecl{{.*}} param
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: AnnotateAttr{{.*}} "method"
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: AnnotateAttr{{.*}} "test"
// CHECK-NEXT: AnnotateAttr{{.*}} "method"

namespace testNamespace {
}
// CHECK-LABEL: NamespaceDecl{{.*}} testNamespace
// CHECK-NEXT: AnnotateAttr{{.*}} "test"

class testClassForward;
// CHECK-LABEL: CXXRecordDecl{{.*}} testClassForward
// CHECK-NEXT: AnnotateAttr{{.*}} "test"

namespace testNamespaceAlias = testNamespace;
// CHECK-LABEL: NamespaceAliasDecl{{.*}} testNamespaceAlias
// CHECK-NOT: AnnotateAttr

using testTypeAlias = testClass2;
// CHECK-LABEL: TypeAliasDecl{{.*}} testTypeAlias
// CHECK: AnnotateAttr{{.*}} "test"

void testCatchVariable() {
  try {
  } catch (int testCatch) {
  }
  testCatchVariable();
}
// CHECK-LABEL: FunctionDecl{{.*}} testCatchVariable
// CHECK: CXXCatchStmt
// CHECK-NEXT: VarDecl{{.*}} testCatch
// CHECK-NEXT: AnnotateAttr{{.*}} "test"

void testLambdaMethod() {
  auto l = [] () { };
  testLambdaMethod();
}
// CHECK-LABEL: FunctionDecl{{.*}} testLambdaMethod
// CHECK: LambdaExpr
// CHECK-NEXT: CXXRecordDecl
// CHECK: CXXMethodDecl{{.*}} operator()
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: AnnotateAttr{{.*}} "test"

#pragma clang attribute pop

#pragma clang attribute push (__attribute__((require_constant_initialization)), apply_to=variable(is_global))

int testCI1 = 1;
// CHECK-LABEL: VarDecl{{.*}} testCI1
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: ConstInitAttr

#pragma clang attribute pop

int testNoCI = 0;
// CHECK-LABEL: VarDecl{{.*}} testNoCI
// CHECK-NEXT: IntegerLiteral
// CHECK-NOT: ConstInitAttr

// Check support for CXX11 style attributes
#pragma clang attribute push ([[noreturn]], apply_to = function)

void testNoReturn();
// CHECK-LABEL: FunctionDecl{{.*}} testNoReturn
// CHECK-NEXT: CXX11NoReturnAttr

#pragma clang attribute pop
