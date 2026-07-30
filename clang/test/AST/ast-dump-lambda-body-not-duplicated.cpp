// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -ast-dump %s \
// RUN: | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -Wno-unused-value \
// RUN: -include-pch %t -ast-dump-all /dev/null \
// RUN: | FileCheck %s

// Make sure that the Stmt * for the body of the LambdaExpr is
// equal to the Stmt * for the body of the call operator.
void Test0() {
  []() {
    return 42;
  };
}

// CHECK: FunctionDecl {{.*}} Test0
//
// CHECK: CXXMethodDecl {{.*}} operator() 'int () const' inline
// CHECK-NEXT: CompoundStmt 0x[[TMP0:.*]]
// CHECK: IntegerLiteral {{.*}} 'int' 42
//
// CHECK: CompoundStmt 0x[[TMP0]]
// Check: IntegerLiteral {{.*}} 'int' 42

void Test1() {
  [](auto x) { return x; };
}

// CHECK: FunctionDecl {{.*}} Test1
//
// CHECK: CXXMethodDecl {{.*}} operator() 'auto (auto) const' inline
// CHECK-NEXT: ParmVarDecl {{.*}} referenced x 'auto'
// CHECK-NEXT: CompoundStmt 0x[[TMP1:.*]]
// CHECK: DeclRefExpr {{.*}} 'x' 'auto'
//
// CHECK: CompoundStmt 0x[[TMP1]]
// CHECK: DeclRefExpr {{.*}} 'x' 'auto'
