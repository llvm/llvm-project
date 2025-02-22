// RUN: %clang_cc1 -std=c++98 %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++23 %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++2c %s -ast-dump | FileCheck %s

namespace cwg2771 { // cwg2771: 18

struct A{
    int a;
    void cwg2771(){
      int* r = &a;
    }
};
// CHECK: CXXMethodDecl{{.+}}cwg2771
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: UnaryOperator
// CHECK-NEXT: MemberExpr
// CHECK-NEXT: CXXThisExpr{{.+}}'cwg2771::A *'

} // namespace cwg2771
