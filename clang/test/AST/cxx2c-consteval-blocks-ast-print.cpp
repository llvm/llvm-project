// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++26 -ast-print %s \
// RUN: | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++26 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -std=c++26 -include-pch %t -ast-print /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

constexpr void f() {}

// CHECK: consteval {
// CHECK: }
consteval {}

// CHECK: consteval {
// CHECK:     f();
// CHECK: }
consteval {
  f();
}

// CHECK: consteval {
// CHECK:     consteval {
// CHECK:         f();
// CHECK:         f();
// CHECK:     }
// CHECK: }
consteval {
  consteval {
    f();
    f();
  }
}

// CHECK: template <typename T> void g() {
// CHECK:     consteval {
// CHECK:         f();
// CHECK:     }
// CHECK: }
template <typename T>
void g() {
  consteval {
    f();
  }
}
