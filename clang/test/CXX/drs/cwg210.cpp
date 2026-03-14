// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-linux-gnu -emit-llvm -disable-llvm-passes -o - -fexceptions -fcxx-exceptions -pedantic-errors | llvm-cxxfilt -n | FileCheck %s --check-prefixes CHECK

#if __cplusplus == 199711L
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvariadic-macros"
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
#pragma clang diagnostic pop
#endif

namespace cwg210 { // cwg210: 2.7
struct B {
  long i;
  B();
  virtual ~B();
};

static_assert(sizeof(B) == 16, "");

struct D : B {
  long j;
  D();
};

static_assert(sizeof(D) == 24, "");

void toss(const B* b) {
  throw *b;
}

// CHECK-LABEL: define {{.*}} void @cwg210::toss(cwg210::B const*)
// CHECK:         %[[EXCEPTION:.*]] = call ptr @__cxa_allocate_exception(i64 16)
// CHECK:         call void @__cxa_throw(ptr %[[EXCEPTION]], ptr @typeinfo for cwg210::B, ptr @cwg210::B::~B())
// CHECK-LABEL: }

} // namespace cwg210
