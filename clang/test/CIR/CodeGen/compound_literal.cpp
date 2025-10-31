// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int foo() {
  int e = (int){1};
  return e;
}

// CIR: %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %[[COMPOUND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, [".compoundliteral", init]
// CIR: %[[VALUE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[VALUE]], %[[COMPOUND]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPOUND]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store{{.*}}  %[[TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_2:.*]] = cir.load{{.*}} %[[INIT]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[TMP_2]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_3:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP_3]] : !s32i

// LLVM: %[[RET:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[COMPOUND:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 1, ptr %[[COMPOUND]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[COMPOUND]], align 4
// LLVM: store i32 %[[TMP]], ptr %[[INIT]], align 4
// LLVM: %[[TMP_2:.*]] = load i32, ptr %[[INIT]], align 4
// LLVM: store i32 %[[TMP_2]], ptr %[[RET]], align 4
// LLVM: %[[TMP_3:.*]] = load i32, ptr %[[RET]], align 4
// LLVM: ret i32 %[[TMP_3]]

// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: %[[COMPOUND:.*]] = alloca i32, align 4
// OGCG: store i32 1, ptr %[[COMPOUND]], align 4
// OGCG: %[[TMP:.*]] = load i32, ptr %[[COMPOUND]], align 4
// OGCG: store i32 %[[TMP]], ptr %[[INIT]], align 4
// OGCG: %[[TMP_2:.*]] = load i32, ptr %[[INIT]], align 4
// OGCG: ret i32 %[[TMP_2]]

void foo2() {
  int _Complex a = (int _Complex) { 1, 2};
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[CL_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, [".compoundliteral"]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.const_complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[CL_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[CL_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP]], %[[A_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM:  %[[A_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[CL_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } { i32 1, i32 2 }, ptr %[[CL_ADDR]], align 4
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[CL_ADDR]], align 4
// LLVM: store { i32, i32 } %[[TMP]], ptr %[[A_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[CL_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[CL_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[CL_ADDR]], i32 0, i32 0
// OGCG: %[[CL_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[CL_ADDR]], i32 0, i32 1
// OGCG: store i32 1, ptr %[[CL_REAL_PTR]], align 4
// OGCG: store i32 2, ptr %[[CL_IMAG_PTR]], align 4
// OGCG: %[[CL_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[CL_ADDR]], i32 0, i32 0
// OGCG: %[[CL_REAL:.*]] = load i32, ptr %[[CL_REAL_PTR]], align 4
// OGCG: %[[CL_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[CL_ADDR]], i32 0, i32 1
// OGCG: %[[CL_IMAG:.*]] = load i32, ptr %[[CL_IMAG_PTR]], align 4
// OGCG: %[[A_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: %[[A_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[CL_REAL]], ptr %[[A_REAL_PTR]], align 4
// OGCG: store i32 %[[CL_IMAG]], ptr %[[A_IMAG_PTR]], align 4

void foo3() {
  typedef int vi4 __attribute__((vector_size(16)));
  auto a = (vi4){10, 20, 30, 40};
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a", init]
// CIR: %[[CL_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, [".compoundliteral", init]
// CIR: %[[VEC:.*]] = cir.const #cir.const_vector<[#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.int<30> : !s32i, #cir.int<40> : !s32i]> : !cir.vector<4 x !s32i>
// CIR: cir.store{{.*}} %[[VEC]], %[[CL_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[CL_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.store{{.*}} %[[TMP]], %[[A_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[CL_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: store <4 x i32> <i32 10, i32 20, i32 30, i32 40>, ptr %[[CL_ADDR]], align 16
// LLVM: %[[TMP:.*]] = load <4 x i32>, ptr %[[CL_ADDR]], align 16
// LLVM: store <4 x i32> %[[TMP]], ptr %[[A_ADDR]], align 16

// OGCG:  %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[CL_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: store <4 x i32> <i32 10, i32 20, i32 30, i32 40>, ptr %[[CL_ADDR]], align 16
// OGCG: %[[TMP:.*]] = load <4 x i32>, ptr %[[CL_ADDR]], align 16
// OGCG: store <4 x i32> %[[TMP]], ptr %[[A_ADDR]], align 16

struct Point {
  int x, y;
};

void foo4() {
  Point p = (Point){5, 10};
}

// CIR-LABEL: @_Z4foo4v
// CIR:   %[[P:.*]] = cir.alloca !rec_Point, !cir.ptr<!rec_Point>, ["p", init]
// CIR:   %[[P_X:.*]] = cir.get_member %[[P]][0] {name = "x"}
// CIR:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:   cir.store{{.*}} %[[FIVE]], %[[P_X]]
// CIR:   %[[P_Y:.*]] = cir.get_member %[[P]][1] {name = "y"}
// CIR:   %[[TEN:.*]] = cir.const #cir.int<10> : !s32i
// CIR:   cir.store{{.*}} %[[TEN]], %[[P_Y]]

// LLVM-LABEL: @_Z4foo4v
// LLVM:   %[[P:.*]] = alloca %struct.Point
// LLVM:   %[[P_X:.*]] = getelementptr %struct.Point, ptr %[[P]], i32 0, i32 0
// LLVM:   store i32 5, ptr %[[P_X]]
// LLVM:   %[[P_Y:.*]] = getelementptr %struct.Point, ptr %[[P]], i32 0, i32 1
// LLVM:   store i32 10, ptr %[[P_Y]]

// OGCG-LABEL: @_Z4foo4v
// OGCG:   %[[P:.*]] = alloca %struct.Point
// OGCG:   call void @llvm.memcpy{{.*}}(ptr{{.*}} %[[P]], ptr{{.*}} @__const._Z4foo4v.p
