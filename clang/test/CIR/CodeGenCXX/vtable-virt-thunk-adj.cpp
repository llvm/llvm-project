// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -fclangir -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=LLVM
// RUN: %clang_cc1 -I%S/Inputs %s -triple x86_64-apple-darwin10 -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=LLVM

#include <typeinfo>

struct Item {
  const std::type_info &ti;
  const char *name;
  void *(*make)();
};

template<typename T> void *make_impl() { return new T; }
template<typename T> constexpr Item item(const char *name) {
  return { typeid(T), name, make_impl<T> };
}


struct A { virtual ~A(); };
struct B { virtual ~B(); };
struct C : virtual A, virtual B {};

extern constexpr Item items[] = {
  item<B>("B"), item<C>("C")
};

// CIR-LABEL: cir.func {{.*}}@_ZTv0_n24_N1CD1Ev
// CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["this", init]
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
// CIR: %[[THIS_CAST:.*]] = cir.cast bitcast %[[THIS_LOAD]] : !cir.ptr<!rec_C> -> !cir.ptr<!u8i>
// CIR: %[[VTBL_PTR:.*]] = cir.vtable.get_vptr %[[THIS_CAST]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.vptr>
// CIR: %[[VTBL_LOAD:.*]] = cir.load {{.*}}%[[VTBL_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR: %[[VTBL_CAST:.*]] = cir.cast bitcast %[[VTBL_LOAD]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<-24> : !s64i
// CIR: %[[OFFSET_STRIDE:.*]] = cir.ptr_stride %[[VTBL_CAST]], %[[OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR: %[[OFFSET_CAST:.*]] = cir.cast bitcast %[[OFFSET_STRIDE]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR: %[[OFFSET_LOAD:.*]] = cir.load align(8) %[[OFFSET_CAST]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[APPLY_OFFSET:.*]] = cir.ptr_stride %[[THIS_CAST]], %[[OFFSET_LOAD]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR: %[[CAST_THIS_BACK:.*]] = cir.cast bitcast %[[APPLY_OFFSET]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_C>
// CIR: cir.call @_ZN1CD1Ev(%[[CAST_THIS_BACK]])
//
// LLVM-LABEL: define{{.*}}@_ZTv0_n24_N1CD1Ev
// LLVM: %[[THIS:.*]] = alloca ptr
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS]]
// LLVM: %[[VTBL_LOAD:.*]] = load ptr, ptr %[[THIS_LOAD]]
// LLVM: %[[OFFSET_STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[VTBL_LOAD]], i64 -24
// LLVM: %[[OFFSET_LOAD:.*]] = load i64, ptr %[[OFFSET_STRIDE]]
// LLVM: %[[APPLY_OFFSET:.*]] = getelementptr {{.*}}i8, ptr %[[THIS_LOAD]], i64 %[[OFFSET_LOAD]]
// LLVM: call void @_ZN1CD1Ev(ptr {{.*}}%[[APPLY_OFFSET]])
//
// CIR-LABEL: cir.func {{.*}}@_ZTv0_n24_N1CD0Ev
// CIR: %[[THIS:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["this", init]
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS]] : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
// CIR: %[[THIS_CAST:.*]] = cir.cast bitcast %[[THIS_LOAD]] : !cir.ptr<!rec_C> -> !cir.ptr<!u8i>
// CIR: %[[VTBL_PTR:.*]] = cir.vtable.get_vptr %[[THIS_CAST]] : !cir.ptr<!u8i> -> !cir.ptr<!cir.vptr>
// CIR: %[[VTBL_LOAD:.*]] = cir.load {{.*}}%[[VTBL_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR: %[[VTBL_CAST:.*]] = cir.cast bitcast %[[VTBL_LOAD]] : !cir.vptr -> !cir.ptr<!u8i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<-24> : !s64i
// CIR: %[[OFFSET_STRIDE:.*]] = cir.ptr_stride %[[VTBL_CAST]], %[[OFFSET]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR: %[[OFFSET_CAST:.*]] = cir.cast bitcast %[[OFFSET_STRIDE]] : !cir.ptr<!u8i> -> !cir.ptr<!s64i>
// CIR: %[[OFFSET_LOAD:.*]] = cir.load align(8) %[[OFFSET_CAST]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[APPLY_OFFSET:.*]] = cir.ptr_stride %[[THIS_CAST]], %[[OFFSET_LOAD]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR: %[[CAST_THIS_BACK:.*]] = cir.cast bitcast %[[APPLY_OFFSET]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_C>
// CIR: cir.call @_ZN1CD0Ev(%[[CAST_THIS_BACK]])
//
// LLVM-LABEL: define{{.*}}@_ZTv0_n24_N1CD0Ev
// LLVM: %[[THIS:.*]] = alloca ptr
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS]]
// LLVM: %[[VTBL_LOAD:.*]] = load ptr, ptr %[[THIS_LOAD]]
// LLVM: %[[OFFSET_STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[VTBL_LOAD]], i64 -24
// LLVM: %[[OFFSET_LOAD:.*]] = load i64, ptr %[[OFFSET_STRIDE]]
// LLVM: %[[APPLY_OFFSET:.*]] = getelementptr {{.*}}i8, ptr %[[THIS_LOAD]], i64 %[[OFFSET_LOAD]]
// LLVM: call void @_ZN1CD0Ev(ptr {{.*}}%[[APPLY_OFFSET]])
