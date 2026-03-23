// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --input-file=%t-ogcg.ll %s -check-prefix=OGCG

void test_sfence(void) {
  // CIR-LABEL: @test_sfence
  // CIR: cir.call_llvm_intrinsic "x86.sse.sfence"  : () -> !void

  // LLVM-LABEL: @test_sfence
  // LLVM: call void @llvm.x86.sse.sfence

  // OGCG-LABEL: @test_sfence
  // OGCG: call void @llvm.x86.sse.sfence
  __builtin_ia32_sfence();
}

void test_lfence(void) {
  // CIR-LABEL: @test_lfence
  // CIR: cir.call_llvm_intrinsic "x86.sse2.lfence"  : () -> !void

  // LLVM-LABEL: @test_lfence
  // LLVM: call void @llvm.x86.sse2.lfence()

  // OGCG-LABEL: @test_lfence
  // OGCG: call void @llvm.x86.sse2.lfence()
  __builtin_ia32_lfence();
}

void test_pause(void) {
  // CIR-LABEL: @test_pause
  // CIR: cir.call_llvm_intrinsic "x86.sse2.pause"  : () -> !void
  
  // LLVM-LABEL: @test_pause
  // LLVM: call void @llvm.x86.sse2.pause()

  // OGCG-LABEL: @test_pause
  // OGCG: call void @llvm.x86.sse2.pause()
  __builtin_ia32_pause();
}

void test_clflush(void* a){
  // CIR-LABEL: test_clflush
  // CIR: cir.call_llvm_intrinsic "x86.sse2.clflush" %{{.*}} : (!cir.ptr<!void, target_address_space(0)>) -> !void
  
  // LLVM-LABEL: @test_clflush
  // LLVM: call void @llvm.x86.sse2.clflush(ptr {{.*}})

  // OGCG-LABEL: @test_clflush
  // OGCG: call void @llvm.x86.sse2.clflush(ptr {{.*}})
  __builtin_ia32_clflush(a);
}

typedef float v4f __attribute__((vector_size(16)));
typedef int   v4i __attribute__((vector_size(16)));

v4i test_convertvector(v4f a) {
  // CIR-LABEL: test_convertvector
  // CIR: cir.cast float_to_int %{{.*}} : !cir.vector<4 x !cir.float> -> !cir.vector<4 x !s32i>

  // LLVM-LABEL: @test_convertvector
  // LLVM: fptosi <4 x float> %{{.*}} to <4 x i32>

  // OGCG-LABEL: @test_convertvector
  // OGCG: fptosi <4 x float> %{{.*}} to <4 x i32>
  return __builtin_convertvector(a, v4i);
}

void foo();
void test_conditional_bzero(void) {
// CIR-LABEL: test_conditional_bzero
// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!s8i x 20>
// CIR: %[[SIZE:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init]
// CIR: %[[ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s8i x 20>> -> !cir.ptr<!s8i>
// CIR: %[[ARR_TO_VOID_PTR:.*]] = cir.cast bitcast %[[ARR_DECAY]] : !cir.ptr<!s8i> -> !cir.ptr<!void>
// CIR: %[[SIZE_LOAD:.*]] = cir.load {{.*}}%[[SIZE]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[SIZE_CAST:.*]] = cir.cast integral %[[SIZE_LOAD]] : !s32i -> !u64i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CIR: cir.libc.memset %[[SIZE_CAST]] bytes at %[[ARR_TO_VOID_PTR]] align(16) to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i
//
// CIR: %[[ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s8i x 20>> -> !cir.ptr<!s8i>
// CIR: %[[ARR_TO_VOID_PTR:.*]] = cir.cast bitcast %[[ARR_DECAY]] : !cir.ptr<!s8i> -> !cir.ptr<!void>
// CIR: %[[SIZE_LOAD:.*]] = cir.load {{.*}}%[[SIZE]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[SIZE_CAST:.*]] = cir.cast integral %[[SIZE_LOAD]] : !s32i -> !u64i
// CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CIR: cir.libc.memset %[[SIZE_CAST]] bytes at %[[ARR_TO_VOID_PTR]] align(16) to %[[ZERO]] : !cir.ptr<!void>, !u8i, !u64i
//
// LLVM-LABEL: @test_conditional_bzero
// LLVM: %[[ARR:.*]] = alloca [20 x i8]
// LLVM: %[[ARR_DECAY:.*]] = getelementptr i8, ptr %[[ARR]], i32 0
// LLVM: %[[SIZE_LOAD:.*]] = load i32, ptr %{{.*}}
// LLVM: %[[SIZE_CAST:.*]] = sext i32 %[[SIZE_LOAD]] to i64
// LLVM: call void @llvm.memset.p0.i64(ptr align 16 %[[ARR_DECAY]], i8 0, i64 %[[SIZE_CAST]], i1 false)
//
// LLVM: %[[ARR_DECAY:.*]] = getelementptr i8, ptr %[[ARR]], i32 0
// LLVM: %[[SIZE_LOAD:.*]] = load i32, ptr %{{.*}}
// LLVM: %[[SIZE_CAST:.*]] = sext i32 %[[SIZE_LOAD]] to i64
// LLVM: call void @llvm.memset.p0.i64(ptr align 16 %[[ARR_DECAY]], i8 0, i64 %[[SIZE_CAST]], i1 false)
//
// OGCG-LABEL: @test_conditional_bzero
// OGCG: %[[ARR:.*]] = alloca [20 x i8]
// OGCG: %[[ARR_DECAY:.*]] = getelementptr inbounds [20 x i8], ptr %[[ARR]], i64 0
// OGCG: %[[SIZE_LOAD:.*]] = load i32, ptr %{{.*}}
// OGCG: %[[SIZE_CAST:.*]] = sext i32 %[[SIZE_LOAD]] to i64
// OGCG: call void @llvm.memset.p0.i64(ptr align 16 %[[ARR_DECAY]], i8 0, i64 %[[SIZE_CAST]], i1 false)
//
// OGCG: %[[ARR_DECAY:.*]] = getelementptr inbounds [20 x i8], ptr %[[ARR]], i64 0
// OGCG: %[[SIZE_LOAD:.*]] = load i32, ptr %{{.*}}
// OGCG: %[[SIZE_CAST:.*]] = sext i32 %[[SIZE_LOAD]] to i64
// OGCG: call void @llvm.memset.p0.i64(ptr align 16 %[[ARR_DECAY]], i8 0, i64 %[[SIZE_CAST]], i1 false)

  char dst[20];
  int _sz = 20, len = 20;
  return (_sz
          ? ((_sz >= len)
              ? __builtin_bzero(dst, len)
              : foo())
          : __builtin_bzero(dst, len));
}
