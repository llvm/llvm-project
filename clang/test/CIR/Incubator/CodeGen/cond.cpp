// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

struct __less {
  inline constexpr bool operator()(const unsigned long& __x, const unsigned long& __y) const {return __x < __y;}
};

const unsigned long&
min(const unsigned long& __a, const unsigned long& __b) {
  return __less()(__b, __a) ? __b : __a;
}

// CHECK: cir.func {{.*}} @_Z3minRKmS0_(%arg0: !cir.ptr<!u64i>
// CHECK:   %0 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__a", init, const] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__b", init, const] {alignment = 8 : i64}
// CHECK:   %2 = cir.alloca !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>, ["__retval"] {alignment = 8 : i64}
// CHECK:   cir.store{{.*}} %arg0, %0 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
// CHECK:   cir.store{{.*}} %arg1, %1 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>
// CHECK:   cir.scope {
// CHECK:     %4 = cir.alloca !rec___less, !cir.ptr<!rec___less>, ["ref.tmp0"] {alignment = 1 : i64}
// CHECK:     %5 = cir.load %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:     %6 = cir.load %0 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:     %7 = cir.call @_ZNK6__lessclERKmS1_(%4, %5, %6) : (!cir.ptr<!rec___less>, !cir.ptr<!u64i>, !cir.ptr<!u64i>) -> !cir.bool
// CHECK:     %8 = cir.ternary(%7, true {
// CHECK:       %9 = cir.load %1 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:       cir.yield %9 : !cir.ptr<!u64i>
// CHECK:     }, false {
// CHECK:       %9 = cir.load %0 : !cir.ptr<!cir.ptr<!u64i>>, !cir.ptr<!u64i>
// CHECK:       cir.yield %9 : !cir.ptr<!u64i>
// CHECK:     }) : (!cir.bool) -> !cir.ptr<!u64i>
// CHECK:     cir.store{{.*}} %8, %2 : !cir.ptr<!u64i>, !cir.ptr<!cir.ptr<!u64i>>

// LLVM-LABEL: define {{.*}} @_Z3minRKmS0_
// LLVM-NOT:     call {{.*}} @_ZN6__lessC1Ev
// LLVM:         call {{.*}} @_ZNK6__lessclERKmS1_
// LLVM:         ret ptr

// OGCG-LABEL: define {{.*}} @_Z3minRKmS0_
// OGCG-NOT:     call {{.*}} @_ZN6__lessC1Ev
// OGCG:         call {{.*}} @_ZNK6__lessclERKmS1_
// OGCG:         ret ptr
