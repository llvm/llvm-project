// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

namespace dr185 { // dr185: 2.7
struct A {
  mutable int value;
  explicit A(int i) : value(i) {}
  void mutate(int i) const { value = i; }
};

int foo() {
  A const& t = A(1);
  A n(t);
  t.mutate(2);
  return n.value;
}

// CHECK-LABEL: define {{.*}} i32 @dr185::foo()
// CHECK:         call void @dr185::A::A(int)(ptr {{[^,]*}} %ref.tmp, {{.*}})
// CHECK:         store ptr %ref.tmp, ptr %t
// CHECK-NOT:     %t =
// CHECK:         [[DR185_T:%.+]] = load ptr, ptr %t
// CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr {{[^,]*}} %n, ptr {{[^,]*}} [[DR185_T]], {{.*}})
// CHECK-LABEL: }
} // namespace dr185
