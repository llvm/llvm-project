// RUN: %clang_cc1 -std=c++17 -triple x86_64-pc-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-pc-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-pc-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct some_struct {
  int a;
  float b;
};

float function() {
  auto[a, b] = some_struct{1, 2.f};

  return a + b;
}

// CIR-LABEL: cir.func dso_local @_Z8functionv() -> !cir.float
// CIR:  %[[RETVAL:.+]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["__retval"]
// CIR:  %[[STRUCT:.+]] = cir.alloca !rec_some_struct, !cir.ptr<!rec_some_struct>, ["", init]
// CIR:  %[[MEMBER_A:.+]] = cir.get_member %[[STRUCT]][0] {name = "a"} : !cir.ptr<!rec_some_struct> -> !cir.ptr<!s32i>
// CIR:  %[[CONST_1:.+]] = cir.const #cir.int<1> : !s32i
// CIR:  cir.store{{.*}} %[[CONST_1]], %[[MEMBER_A]]
// CIR:  %[[MEMBER_B:.+]] = cir.get_member %[[STRUCT]][1] {name = "b"} : !cir.ptr<!rec_some_struct> -> !cir.ptr<!cir.float>
// CIR:  %[[TWO_FP:.+]] = cir.const #cir.fp<2.000000e+00> : !cir.float
// CIR:  cir.store{{.*}} %[[TWO_FP]], %[[MEMBER_B]]
// CIR:  %[[MEMBER_A:.+]] = cir.get_member %[[STRUCT]][0] {name = "a"} : !cir.ptr<!rec_some_struct> -> !cir.ptr<!s32i>
// CIR:  %[[LOAD_A:.+]] = cir.load align(4) %[[MEMBER_A]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[CAST_A:.+]] = cir.cast(int_to_float, %[[LOAD_A]] : !s32i), !cir.float
// CIR:  %[[MEMBER_B:.+]] = cir.get_member %[[STRUCT]][1] {name = "b"} : !cir.ptr<!rec_some_struct> -> !cir.ptr<!cir.float>
// CIR:  %[[LOAD_B:.+]] = cir.load align(4) %[[MEMBER_B]] : !cir.ptr<!cir.float>, !cir.float
// CIR:  %[[ADD:.+]] = cir.binop(add, %[[CAST_A]], %[[LOAD_B]]) : !cir.float
// CIR:  cir.store %[[ADD]], %[[RETVAL]] : !cir.float, !cir.ptr<!cir.float>
// CIR:  %[[RET:.+]] = cir.load %[[RETVAL]] : !cir.ptr<!cir.float>, !cir.float
// CIR:  cir.return %[[RET]] : !cir.float

// LLVM-LABEL: define dso_local float @_Z8functionv()
// LLVM:  %[[RETVAL:.+]] = alloca float, i64 1
// LLVM:  %[[STRUCT:.+]] = alloca %struct.some_struct, i64 1
// LLVM:  %[[GEP_A:.+]] = getelementptr %struct.some_struct, ptr %[[STRUCT]], i32 0, i32 0
// LLVM:  store i32 1, ptr %[[GEP_A]]
// LLVM:  %[[GEP_B:.+]] = getelementptr %struct.some_struct, ptr %[[STRUCT]], i32 0, i32 1
// LLVM:  store float 2.000000e+00, ptr %[[GEP_B]]
// LLVM:  %[[GEP_A:.+]] = getelementptr %struct.some_struct, ptr %[[STRUCT]], i32 0, i32 0
// LLVM:  %[[LOAD_A:.+]] = load i32, ptr %[[GEP_A]]
// LLVM:  %[[CAST_A:.+]] = sitofp i32 %[[LOAD_A]] to float
// LLVM:  %[[GEP_B:.+]] = getelementptr %struct.some_struct, ptr %[[STRUCT]], i32 0, i32 1
// LLVM:  %[[LOAD_B:.+]] = load float, ptr %[[GEP_B]]
// LLVM:  %[[ADD:.+]] = fadd float %[[CAST_A]], %[[LOAD_B]]
// LLVM:  store float %[[ADD]], ptr %[[RETVAL]]
// LLVM:  %[[RET:.+]] = load float, ptr %[[RETVAL]]
// LLVM:  ret float %[[RET]]

// OGCG: @__const._Z8functionv.{{.*}} = private unnamed_addr constant %struct.some_struct { i32 1, float 2.000000e+00 }
// OGCG-LABEL: define dso_local noundef float @_Z8functionv()
// OGCG:  %[[STRUCT:.+]] = alloca %struct.some_struct
// OGCG:  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[STRUCT]], ptr align 4 @__const._Z8functionv.{{.*}}, i64 8, i1 false)
// OGCG:  %[[GEP_A:.+]] = getelementptr inbounds nuw %struct.some_struct, ptr %[[STRUCT]], i32 0, i32 0
// OGCG:  %[[LOAD_A:.+]] = load i32, ptr %[[GEP_A]]
// OGCG:  %[[CAST_A:.+]] = sitofp i32 %[[LOAD_A]] to float
// OGCG:  %[[GEP_B:.+]] = getelementptr inbounds nuw %struct.some_struct, ptr %[[STRUCT]], i32 0, i32 1
// OGCG:  %[[LOAD_B:.+]] = load float, ptr %[[GEP_B]]
// OGCG:  %[[ADD:.+]] = fadd float %[[CAST_A]], %[[LOAD_B]]
// OGCG:  ret float %[[ADD]]
