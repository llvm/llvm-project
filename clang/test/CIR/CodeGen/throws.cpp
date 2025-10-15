// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void rethrow() {
  throw;
}

// CIR: cir.throw
// CIR: cir.unreachable

// LLVM: call void @__cxa_rethrow()
// LLVM: unreachable

// OGCG: call void @__cxa_rethrow()
// OGCG: unreachable

int rethrow_from_block(int a, int b) {
  if (b == 0)
    throw;
  return a / b;
}

// CIR:  %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR:  %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:  %[[RES_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:  cir.store %{{.*}}, %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:  cir.store %{{.*}}, %[[B_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:  cir.scope {
// CIR:    %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:    %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR:    %[[IS_B_ZERO:.*]] = cir.cmp(eq, %[[TMP_B]], %[[CONST_0]]) : !s32i, !cir.bool
// CIR:    cir.if %[[IS_B_ZERO]] {
// CIR:      cir.throw
// CIR:      cir.unreachable
// CIR:    }
// CIR:  }
// CIR:  %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:  %[[DIV_A_B:.*]] = cir.binop(div, %[[TMP_A:.*]], %[[TMP_B:.*]]) : !s32i
// CIR:  cir.store %[[DIV_A_B]], %[[RES_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:  %[[RESULT:.*]] = cir.load %[[RES_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:  cir.return %[[RESULT]] : !s32i

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[RES_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 %{{.*}}, ptr %[[A_ADDR]], align 4
// LLVM: store i32 %{{.*}}, ptr %[[B_ADDR]], align 4
// LLVM: br label %[[CHECK_COND:.*]]
// LLVM: [[CHECK_COND]]:
// LLVM:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:  %[[IS_B_ZERO:.*]] = icmp eq i32 %[[TMP_B]], 0
// LLVM:  br i1 %[[IS_B_ZERO]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
// LLVM: [[IF_THEN]]:
// LLVM:  call void @__cxa_rethrow()
// LLVM:  unreachable
// LLVM: [[IF_ELSE]]:
// LLVM:  br label %[[IF_END:.*]]
// LLVM: [[IF_END]]:
// LLVM:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:  %[[DIV_A_B:.*]] = sdiv i32 %[[TMP_A]], %[[TMP_B]]
// LLVM:  store i32 %[[DIV_A_B]], ptr %[[RES_ADDR]], align 4
// LLVM:  %[[RESULT:.*]] = load i32, ptr %[[RES_ADDR]], align 4
// LLVM:  ret i32 %[[RESULT]]

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 %{{.*}}, ptr %[[A_ADDR]], align 4
// OGCG: store i32 %{{.*}}, ptr %[[B_ADDR]], align 4
// OGCG: %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG: %[[IS_B_ZERO:.*]] = icmp eq i32 %[[TMP_B]], 0
// OGCG: br i1 %[[IS_B_ZERO]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// OGCG: [[IF_THEN]]:
// OGCG:  call void @__cxa_rethrow()
// OGCG:  unreachable
// OGCG: [[IF_END]]:
// OGCG:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG:  %[[DIV_A_B:.*]] = sdiv i32 %[[TMP_A]], %[[TMP_B]]
// OGCG:  ret i32 %[[DIV_A_B]]

void throw_scalar() { 
  throw 1;
}

// CIR: %[[EXCEPTION_ADDR:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR: %[[EXCEPTION_VALUE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[EXCEPTION_VALUE]], %[[EXCEPTION_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.throw %[[EXCEPTION_ADDR]] : !cir.ptr<!s32i>, @_ZTIi
// CIR: cir.unreachable

// LLVM: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// LLVM: store i32 1, ptr %[[EXCEPTION_ADDR]], align 16
// LLVM: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIi, ptr null)
// LLVM: unreachable

// OGCG: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// OGCG: store i32 1, ptr %[[EXCEPTION_ADDR]], align 16
// OGCG: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIi, ptr null)
// OGCG: unreachable

void paren_expr() { (throw 0, 1 + 2); }

// CIR:   %[[EXCEPTION_ADDR:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR:   %[[EXCEPTION_VALUE:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[EXCEPTION_VALUE]], %[[EXCEPTION_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.throw %[[EXCEPTION_ADDR]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:   cir.unreachable
// CIR: ^bb1:
// CIR:   %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   %[[CONST_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR:   %[[ADD:.*]] = cir.binop(add, %[[CONST_1]], %[[CONST_2]]) nsw : !s32i

// LLVM: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// LLVM: store i32 0, ptr %[[EXCEPTION_ADDR]], align 16
// LLVM: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIi, ptr null)

// OGCG: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// OGCG: store i32 0, ptr %[[EXCEPTION_ADDR]], align 16
// OGCG: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIi, ptr null)

void throw_complex_expr() {
  throw __builtin_complex(1.1f, 2.2f);
}

// CIR: %[[EXCEPTION_ADDR:.*]] = cir.alloc.exception 8 -> !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[EXCEPTION_VALUE:.*]] = cir.const #cir.const_complex<#cir.fp<1.100000e+00> : !cir.float, #cir.fp<2.200000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[EXCEPTION_VALUE]], %[[EXCEPTION_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: cir.throw %[[EXCEPTION_ADDR]] : !cir.ptr<!cir.complex<!cir.float>>, @_ZTICf
// CIR: cir.unreachable

// LLVM: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 8)
// LLVM: store { float, float } { float 0x3FF19999A0000000, float 0x40019999A0000000 }, ptr %[[EXCEPTION_ADDR]], align 16
// LLVM: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTICf, ptr null)

// OGCG: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 8)
// OGCG: %[[EXCEPTION_REAL:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[EXCEPTION_ADDR]], i32 0, i32 0
// OGCG: %[[EXCEPTION_IMAG:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[EXCEPTION_ADDR]], i32 0, i32 1
// OGCG: store float 0x3FF19999A0000000, ptr %[[EXCEPTION_REAL]], align 16
// OGCG: store float 0x40019999A0000000, ptr %[[EXCEPTION_IMAG]], align 4
// OGCG: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTICf, ptr null)

void throw_vector_type() {
  typedef int vi4 __attribute__((vector_size(16)));
  vi4 a;
  throw a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[EXCEPTION_ADDR:.*]] = cir.alloc.exception 16 -> !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.store{{.*}} %[[TMP_A]], %[[EXCEPTION_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.throw %[[EXCEPTION_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, @_ZTIDv4_i
// CIR: cir.unreachable

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 16)
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: store <4 x i32> %[[TMP_A]], ptr %[[EXCEPTION_ADDR]], align 16
// LLVM: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIDv4_i, ptr null)

// OGCG: %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 16)
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: store <4 x i32> %[[TMP_A]], ptr %[[EXCEPTION_ADDR]], align 16
// OGCG: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIDv4_i, ptr null)
// OGCG: unreachable

void throw_ext_vector_type() {
  typedef int vi4 __attribute__((ext_vector_type(4)));
  vi4 a;
  throw a;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>, ["a"]
// CIR: %[[EXCEPTION_ADDR:.*]] = cir.alloc.exception 16 -> !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, !cir.vector<4 x !s32i>
// CIR: cir.store{{.*}} %[[TMP_A]], %[[EXCEPTION_ADDR]] : !cir.vector<4 x !s32i>, !cir.ptr<!cir.vector<4 x !s32i>>
// CIR: cir.throw %[[EXCEPTION_ADDR]] : !cir.ptr<!cir.vector<4 x !s32i>>, @_ZTIDv4_i
// CIR: cir.unreachable

// LLVM: %[[A_ADDR:.*]] = alloca <4 x i32>, i64 1, align 16
// LLVM: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 16)
// LLVM: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// LLVM: store <4 x i32> %[[TMP_A]], ptr %[[EXCEPTION_ADDR]], align 16
// LLVM: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIDv4_i, ptr null)

// OGCG: %[[A_ADDR:.*]] = alloca <4 x i32>, align 16
// OGCG: %[[EXCEPTION_ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 16)
// OGCG: %[[TMP_A:.*]] = load <4 x i32>, ptr %[[A_ADDR]], align 16
// OGCG: store <4 x i32> %[[TMP_A]], ptr %[[EXCEPTION_ADDR]], align 16
// OGCG: call void @__cxa_throw(ptr %[[EXCEPTION_ADDR]], ptr @_ZTIDv4_i, ptr null)
// OGCG: unreachable
