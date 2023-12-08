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

// CHECK: analyzer-display-progress.cpp f() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp g() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp h() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp SomeStruct::f() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp SomeOtherStruct::f() : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(int) : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(float, ::SomeStruct) : {{[0-9]+}}
// CHECK: analyzer-display-progress.cpp ns::SomeStruct::f(float, SomeStruct) : {{[0-9]+}}
