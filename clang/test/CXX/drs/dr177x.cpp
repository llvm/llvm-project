// RUN: %clang_cc1 -std=c++98 %s -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s --check-prefixes=CHECK,CXX11
// RUN: %clang_cc1 -std=c++14 %s -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s --check-prefixes=CHECK,CXX11,CXX14
// RUN: %clang_cc1 -std=c++1z %s -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s --check-prefixes=CHECK,CXX11,CXX14

namespace dr1772 { // dr1772: 14
  // __func__ in a lambda should name operator(), not the containing function.
  // CHECK: NamespaceDecl{{.+}}dr1772
#if __cplusplus >= 201103L
  auto x = []() { __func__; };
  // CXX11: LambdaExpr
  // CXX11: CXXRecordDecl
  // CXX11: CXXMethodDecl{{.+}} operator() 'void () const'
  // CXX11-NEXT: CompoundStmt
  // CXX11-NEXT: PredefinedExpr{{.+}} 'const char[11]' lvalue __func__
  // CXX11-NEXT: StringLiteral{{.+}} 'const char[11]' lvalue "operator()"

  void func() {
    // CXX11: FunctionDecl{{.+}} func
  (void)[]() { __func__; };
  // CXX11-NEXT: CompoundStmt
  // CXX11: LambdaExpr
  // CXX11: CXXRecordDecl
  // CXX11: CXXMethodDecl{{.+}} operator() 'void () const'
  // CXX11-NEXT: CompoundStmt
  // CXX11-NEXT: PredefinedExpr{{.+}} 'const char[11]' lvalue __func__
  // CXX11-NEXT: StringLiteral{{.+}} 'const char[11]' lvalue "operator()"
  }
#endif // __cplusplus >= 201103L
}

namespace dr1779 { // dr1779: 14
  // __func__ in a function template, member function template, or generic
  //  lambda should have a dependent type.
  // CHECK: NamespaceDecl{{.+}}dr1779

  template<typename T>
  void FuncTemplate() {
    __func__;
    // CHECK: FunctionDecl{{.+}} FuncTemplate
    // CHECK-NEXT: CompoundStmt
    // CHECK-NEXT: PredefinedExpr{{.+}} '<dependent type>' lvalue __func__
  }

  template<typename T>
  class ClassTemplate {
    // CHECK: ClassTemplateDecl{{.+}} ClassTemplate
    void MemFunc() {
      // CHECK: CXXMethodDecl{{.+}} MemFunc 'void ()'
      // CHECK-NEXT: CompoundStmt
      // CHECK-NEXT: PredefinedExpr{{.+}} '<dependent type>' lvalue __func__
      __func__;
    }
    void OutOfLineMemFunc();
  };

  template <typename T> void ClassTemplate<T>::OutOfLineMemFunc() {
    // CHECK: CXXMethodDecl{{.+}}parent{{.+}} OutOfLineMemFunc 'void ()'
    // CHECK-NEXT: CompoundStmt
    // CHECK-NEXT: PredefinedExpr{{.+}} '<dependent type>' lvalue __func__
    __func__;
  }

#if __cplusplus >= 201402L
  void contains_generic_lambda() {
    // CXX14: FunctionDecl{{.+}}contains_generic_lambda
    // CXX14: LambdaExpr
    // CXX14: CXXRecordDecl
    // CXX14: CXXMethodDecl{{.+}} operator() 'auto (auto) const'
    // CXX14-NEXT: ParmVarDecl
    // CXX14-NEXT: CompoundStmt
    // CXX14-NEXT: PredefinedExpr{{.+}} '<dependent type>' lvalue __func__
    (void)[](auto x) {
      __func__;
    };
  }
#endif // __cplusplus >= 201402L
}
