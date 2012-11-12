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
  
  // CHECK: void f()
  void f() {
    // CHECK: bar<int> tmp =
    // CHECK: CXXDefaultArgExpr{{.*}}'int'
    bar<int> tmp;
  }
}
