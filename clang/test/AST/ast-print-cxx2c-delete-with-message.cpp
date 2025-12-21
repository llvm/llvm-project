// Without serialization:
// RUN: %clang_cc1 -ast-print %s | FileCheck %s
//
// With serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -include-pch %t -ast-print  /dev/null | FileCheck %s

// CHECK: struct S {
struct S {
  // CHECK-NEXT: void a() = delete("foo");
  void a() = delete("foo");

  // CHECK-NEXT: template <typename T> T b() = delete("bar");
  template <typename T> T b() = delete("bar");
};

// CHECK: void c() = delete("baz");
void c() = delete("baz");
