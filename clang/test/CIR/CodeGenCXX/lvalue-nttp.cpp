// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

template<auto &V> void templ() {
  V = 1;
}
int i;
template void templ<i>();
// CIR: cir.func{{.*}}@_Z5templITnRDaL_Z1iEEvv()
// CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-NEXT: %[[GLOB:.*]] = cir.get_global @i
// CIR-NEXT: cir.store{{.*}} %[[ONE]], %[[GLOB]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}}@_Z5templITnRDaL_Z1iEEvv()
// LLVM: store i32 1, ptr @i

float f;
template void templ<f>();
// CIR: cir.func{{.*}}@_Z5templITnRDaL_Z1fEEvv()
// CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-NEXT: %[[CAST:.*]] = cir.cast int_to_float %[[ONE]] : !s32i -> !cir.float
// CIR-NEXT: %[[GLOB:.*]] = cir.get_global @f
// CIR-NEXT: cir.store{{.*}} %[[CAST]], %[[GLOB]] : !cir.float, !cir.ptr<!cir.float>
//
// LLVM: define{{.*}}@_Z5templITnRDaL_Z1fEEvv()
// LLVM: store float 1{{.*}}, ptr @f

struct Struct{Struct(); Struct(int);};
Struct s;
template void templ<s>();
// CIR: cir.func{{.*}}@_Z5templITnRDaL_Z1sEEvv()
// CIR-NEXT: %[[TEMP:.*]] = cir.alloca !rec_Struct, !cir.ptr<!rec_Struct>
// CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-NEXT: cir.call @_ZN6StructC1Ei(%[[TEMP]], %[[ONE]])
// CIR-NEXT: %[[GLOB:.*]] = cir.get_global @s : !cir.ptr<!rec_Struct>
// CIR-NEXT: cir.call @_ZN6StructaSEOS_(%[[GLOB]], %[[TEMP]])

// LLVM: define{{.*}}@_Z5templITnRDaL_Z1sEEvv()
// LLVM: %[[TEMP:.*]] = alloca %struct.Struct
// LLVM: call void @_ZN6StructC1Ei(ptr{{.*}} %[[TEMP]], i32{{.*}} 1)
