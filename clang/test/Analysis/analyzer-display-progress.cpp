// RUN: %clang_analyze_cc1 -analyzer-display-progress %s 2>&1 | FileCheck %s

void f() {};
void g() {};
void h() {}

struct SomeStruct {
  void f() {}
};

struct SomeOtherStruct {
  void f() {}
};

namespace ns {
  struct SomeStruct {
    void f(int) {}
    void f(float, ::SomeStruct) {}
    void f(float, SomeStruct) {}
  };
}

// CHECK: analyzer-display-progress.cpp f()
// CHECK: analyzer-display-progress.cpp g()
// CHECK: analyzer-display-progress.cpp h()
// CHECK: analyzer-display-progress.cpp SomeStruct::f()
// CHECK: analyzer-display-progress.cpp SomeOtherStruct::f()
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(int)
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(float, ::SomeStruct)
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(float, struct ns::SomeStruct)
