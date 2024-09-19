// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

#if __cplusplus == 199711L
#define NOTHROW throw()
#else
#define NOTHROW noexcept(true)
#endif

namespace std {
struct type_info {
  const char* name() const NOTHROW;
}; 
}

namespace cwg492 { // cwg492: 2.7

void f() {
  typeid(int).name();
  typeid(const int).name();
  typeid(volatile int).name();
  typeid(const volatile int).name();
}

} // namespace cwg492

// CHECK-LABEL: define {{.*}} void @cwg492::f()()
// CHECK:         {{.*}} = call {{.*}} @std::type_info::name() const({{.*}} @typeinfo for int)
// CHECK-NEXT:    {{.*}} = call {{.*}} @std::type_info::name() const({{.*}} @typeinfo for int)
// CHECK-NEXT:    {{.*}} = call {{.*}} @std::type_info::name() const({{.*}} @typeinfo for int)
// CHECK-NEXT:    {{.*}} = call {{.*}} @std::type_info::name() const({{.*}} @typeinfo for int)
// CHECK-LABEL: }
