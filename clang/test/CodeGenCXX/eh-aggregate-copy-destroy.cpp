// Check that in case of copying an array of memcpy-able objects, their
// destructors will be called if an exception is thrown.
//
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fexceptions -fcxx-exceptions -O0 -fno-elide-constructors -std=c++98 -emit-llvm %s -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK98 %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fexceptions -fcxx-exceptions -O0 -fno-elide-constructors -std=c++11 -emit-llvm %s -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK11 %s

struct ImplicitCopy {
  int x;
  ImplicitCopy() { x = 10; }
  ~ImplicitCopy() { x = 20; }
};

struct ThrowCopy {
  ThrowCopy() {}
  ThrowCopy(const ThrowCopy &) { throw 1; }
};

struct Container {
  ImplicitCopy b[2];
  ThrowCopy c;
};

int main () {
  try {
    Container c1;
    // CHECK-LABEL: main
    // CHECK-NOT: call void @_ZN9ThrowCopyC1ERKS_
    // CHECK: invoke void @_ZN9ThrowCopyC1ERKS_
    // CHECK98: invoke void @_ZN12ImplicitCopyD1Ev
    // CHECK11: call void @_ZN12ImplicitCopyD1Ev
    Container c2(c1);
  }
  catch (...) {
    return 1;
  }

  return 0;
}

