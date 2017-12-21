// RUN: %clang_cc1 -ast-dump %s 2>&1 | FileCheck %s

// This is a wacky test to ensure that we're actually instantiating
// the default arguments of the constructor when the function type is
// otherwise non-dependent.
namespace PR6733 {
  template <class T>
  class bar {
  public: enum { kSomeConst = 128 };
    bar(int x = kSomeConst) {}
  };
  
  // CHECK: FunctionDecl{{.*}}f 'void (void)'
  void f() {
    // CHECK: VarDecl{{.*}}tmp 'bar<int>'
    // CHECK: CXXDefaultArgExpr{{.*}}'int'
    bar<int> tmp;
  }
}
