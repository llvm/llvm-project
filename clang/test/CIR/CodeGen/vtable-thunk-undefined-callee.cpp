// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -O1 -fno-rtti -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -O1 -fno-rtti -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -O1 -fno-rtti -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

namespace ConstructionVTableThunk {
struct Base {
  virtual int f(int x);
};
struct Left : virtual Base {};
struct Right : virtual Base {
  int f(int x) override;
};
struct Middle : Left, Right {};
struct Derived : Middle {
  int f(int x) override;
};
int Derived::f(int x) { return x; }
} // namespace ConstructionVTableThunk

// The Derived vtable's this-adjusting slot points at the thunk below.
// CIR: cir.global "private" external @_ZTVN23ConstructionVTableThunk7DerivedE = #cir.vtable<{{{.*}}#cir.global_view<@_ZThn8_N23ConstructionVTableThunk7Derived1fEi> : !cir.ptr<!u8i>

// CIR-LABEL: cir.func {{.*}} @_ZThn8_N23ConstructionVTableThunk7Derived1fEi(%arg0: !cir.ptr<
// CIR:   %[[THIS:.*]] = cir.load
// CIR:   %[[CAST:.*]] = cir.cast bitcast %[[THIS]] : !cir.ptr<{{.*}}> -> !cir.ptr<!u8i>
// CIR:   %[[OFFSET:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:   %[[ADJUSTED:.*]] = cir.ptr_stride %[[CAST]], %[[OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[RESULT:.*]] = cir.cast bitcast %[[ADJUSTED]] : !cir.ptr<!u8i> -> !cir.ptr<
// CIR:   cir.call @_ZN23ConstructionVTableThunk7Derived1fEi(%[[RESULT]]
// CIR:   cir.return

// Right::f, materialized on demand during thunk emission, lands at module scope.
// CIR: cir.func private @_ZN23ConstructionVTableThunk5Right1fEi

// LLVM-LABEL: define {{.*}} i32 @_ZThn8_N23ConstructionVTableThunk7Derived1fEi(ptr{{.*}}, i32{{.*}})
// LLVM:   %[[THIS:.*]] = load ptr, ptr
// LLVM:   %[[ADJ:.*]] = getelementptr {{.*}}i8, ptr %[[THIS]], i64 -8
// LLVM:   %[[ARG:.*]] = load i32, ptr
// LLVM:   {{.*}}call {{.*}} i32 @_ZN23ConstructionVTableThunk7Derived1fEi(ptr{{.*}} %[[ADJ]], i32{{.*}} %[[ARG]])
