// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

#if __cplusplus == 199711L
#define NOTHROW throw()
#else
#define NOTHROW noexcept(true)
#endif

namespace cwg392 { // cwg392: 2.8

struct A {
  operator bool() NOTHROW;
};

class C {
public:
  C() NOTHROW;
  ~C() NOTHROW;
  A& get() NOTHROW { return p; }
private:
  A p;
};

void f()
{
  if (C().get()) {}
}

} // namespace cwg392

// CHECK-LABEL: define {{.*}} void @cwg392::f()()
// CHECK:         call {{.*}} i1 @cwg392::A::operator bool()
// CHECK:         call void @cwg392::C::~C()
// CHECK-LABEL: }
