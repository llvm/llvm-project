// RUN: %clang_cc1 -analyze -analyzer-display-progress %s 2>&1 | FileCheck %s

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
    void f() {}
  };
}

// CHECK: analyze_display_progress.cpp f
// CHECK: analyze_display_progress.cpp g
// CHECK: analyze_display_progress.cpp h
// CHECK: analyze_display_progress.cpp SomeStruct::f
// CHECK: analyze_display_progress.cpp SomeOtherStruct::f
// CHECK: analyze_display_progress.cpp ns::SomeStruct::f
