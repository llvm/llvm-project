// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct Base { int x; };

void Field() {
  const int &r = Base().x;
}
// CIR-LABEL: cir.func {{.*}}@_Z5Fieldv() 
// CIR: %[[TEMP_ALLOCA:.*]] = cir.alloca !rec_Base, !cir.ptr<!rec_Base>, ["ref.tmp0"]
// CIR: %[[R_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const]
// CIR:  %[[GET_MEM:.*]] = cir.get_member %[[TEMP_ALLOCA]][0] {name = "x"} : !cir.ptr<!rec_Base> -> !cir.ptr<!s32i>
// CIR:  cir.store align(8) %[[GET_MEM]], %[[R_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM-LABEL: define {{.*}}@_Z5Fieldv()
// LLVM-DAG: %[[TEMP_ALLOCA:.*]] = alloca %struct.Base
// LLVM-DAG: %[[R_ALLOCA:.*]] = alloca ptr
// LLVM: %[[GET_MEM:.*]] = getelementptr inbounds nuw %struct.Base, ptr %[[TEMP_ALLOCA]], i32 0, i32 0
// LLVM: store ptr %[[GET_MEM]], ptr %[[R_ALLOCA]], align 8

void MemPtr(int Base::*mp) {
  const int &r = Base().*mp;
}
// CIR-LABEL: cir.func {{.*}}@_Z6MemPtrM4Basei
// CIR: %[[MP_ALLOCA:.*]] = cir.alloca !s64i, !cir.ptr<!s64i>, ["mp", init] {alignment = 8 : i64}
// CIR: %[[TEMP_ALLOCA:.*]] = cir.alloca !rec_Base, !cir.ptr<!rec_Base>, ["ref.tmp0"] {alignment = 4 : i64}
// CIR: %[[R_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const] {alignment = 8 : i64}
// CIR: %[[ARG_LOAD:.*]] = cir.load align(8) %[[MP_ALLOCA]] : !cir.ptr<!s64i>, !s64i
// CIR: %[[TEMP_LOAD:.*]] = cir.cast bitcast %[[TEMP_ALLOCA]] : !cir.ptr<!rec_Base> -> !cir.ptr<!s8i>
// CIR: %[[STRIDE:.*]] = cir.ptr_stride %[[TEMP_LOAD]], %[[ARG_LOAD]] : (!cir.ptr<!s8i>, !s64i) -> !cir.ptr<!s8i>
// CIR: %[[TO_INT:.*]] = cir.cast bitcast %[[STRIDE:.*]] : !cir.ptr<!s8i> -> !cir.ptr<!s32i>
// CIR: cir.store align(8) %[[TO_INT]], %[[R_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM-LABEL: define {{.*}}@_Z6MemPtrM4Basei
// LLVM: %[[MP_ALLOCA:.*]] = alloca i64
// LLVM-DAG: %[[TEMP_ALLOCA:.*]] = alloca %struct.Base
// LLVM-DAG: %[[R_ALLOCA:.*]] = alloca ptr
// LLVM: %[[ARG_LOAD:.*]] = load i64, ptr %[[MP_ALLOCA]], align 8
// LLVM: %[[STRIDE:.*]] = getelementptr {{.*}}i8, ptr %[[TEMP_ALLOCA]], i64 %[[ARG_LOAD]]
// LLVM: store ptr %[[STRIDE]], ptr %[[R_ALLOCA]], align 8

struct Derived : Base {};
void DerivedToBase() {
  const int &r = Derived().x;
}
// CIR-LABEL: cir.func {{.*}}@_Z13DerivedToBasev()
// CIR: %[[TEMP_ALLOCA:.*]] = cir.alloca !rec_Derived, !cir.ptr<!rec_Derived>, ["ref.tmp0"] {alignment = 4 : i64}
// CIR: %[[R_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const] {alignment = 8 : i64}
// CIR: %[[BASE:.*]] = cir.base_class_addr %[[TEMP_ALLOCA]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR: %[[GET_MEM:.*]] = cir.get_member %[[BASE]][0] {name = "x"} : !cir.ptr<!rec_Base> -> !cir.ptr<!s32i>
// CIR: cir.store align(8) %[[GET_MEM]], %[[R_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM-LABEL: define {{.*}}@_Z13DerivedToBasev
// LLVM-DAG: %[[TEMP_ALLOCA:.*]] = alloca %struct.Derived
// LLVM-DAG: %[[R_ALLOCA:.*]] = alloca ptr
// LLVM: %[[GET_MEM:.*]] = getelementptr inbounds nuw %struct.Base, ptr %[[TEMP_ALLOCA]], i32 0, i32 0
// LLVM: store ptr %[[GET_MEM]], ptr %[[R_ALLOCA]], align 8
