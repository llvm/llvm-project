// RUN: %clang_cc1 -std=c++23 %s -ast-dump | FileCheck --check-prefixes=CXX23 %s

namespace cwg2771 { // cwg2771: 18

struct A{
    int a;
    void cwg2771(){
      int* r = &a;
    }
};
// CXX23: CXXMethodDecl{{.+}}cwg2771
// CXX23-NEXT: CompoundStmt
// CXX23-NEXT: DeclStmt
// CXX23-NEXT: VarDecl
// CXX23-NEXT: UnaryOperator
// CXX23-NEXT: MemberExpr
// CXX23-NEXT: CXXThisExpr{{.+}}'cwg2771::A *'

} // namespace cwg2771
