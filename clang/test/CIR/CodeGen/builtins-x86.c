// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512fp16 -target-feature +avx512bf16 -target-feature +avx512vl -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512fp16 -target-feature +avx512bf16 -target-feature +avx512vl -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512fp16 -target-feature +avx512bf16 -target-feature +avx512vl -emit-llvm %s -o %t-ogcg.ll
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

typedef float    v4f  __attribute__((vector_size(16)));
typedef int      v4i  __attribute__((vector_size(16)));
typedef double   v2d  __attribute__((vector_size(16)));
typedef _Float16 v32h __attribute__((vector_size(64)));
typedef __bf16   v8bf __attribute__((vector_size(16)));

v4f test_cmpeqps(v4f a, v4f b) {
  // CIR-LABEL: @test_cmpeqps
  // CIR: cir.vec.cmp(eq, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_cmpeqps
  // LLVM: fcmp oeq <4 x float>

  // OGCG-LABEL: @test_cmpeqps
  // OGCG: fcmp oeq <4 x float>
  return __builtin_ia32_cmpeqps(a, b);
}

v4f test_cmpltps(v4f a, v4f b) {
  // CIR-LABEL: @test_cmpltps
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_cmpltps
  // LLVM: fcmp olt <4 x float>

  // OGCG-LABEL: @test_cmpltps
  // OGCG: fcmp olt <4 x float>
  return __builtin_ia32_cmpltps(a, b);
}

v4f test_cmpleps(v4f a, v4f b) {
  // CIR-LABEL: @test_cmpleps
  // CIR: cir.vec.cmp(le, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_cmpleps
  // LLVM: fcmp ole <4 x float>

  // OGCG-LABEL: @test_cmpleps
  // OGCG: fcmp ole <4 x float>
  return __builtin_ia32_cmpleps(a, b);
}

v4f test_cmpunordps(v4f a, v4f b) {
  // CIR-LABEL: @test_cmpunordps
  // CIR: cir.vec.cmp(uno, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_cmpunordps
  // LLVM: fcmp uno <4 x float>

  // OGCG-LABEL: @test_cmpunordps
  // OGCG: fcmp uno <4 x float>
  return __builtin_ia32_cmpunordps(a, b);
}

v4f test_cmpordps(v4f a, v4f b) {
  // CIR-LABEL: @test_cmpordps
  // CIR: cir.vec.cmp(uno, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.not %{{.*}} : !cir.vector<4 x !s32i>

  // LLVM-LABEL: @test_cmpordps
  // LLVM: fcmp uno <4 x float>
  // LLVM: xor <4 x i32> %{{.*}}, splat (i32 -1)

  // OGCG-LABEL: @test_cmpordps
  // OGCG: fcmp ord <4 x float>
  return __builtin_ia32_cmpordps(a, b);
}

v4f test_cmpneqps(v4f a, v4f b) {
  // CIR-LABEL: @test_cmpneqps
  // CIR: cir.vec.cmp(ne, %{{.*}}, %{{.*}}) : !cir.vector<4 x !cir.float>, !cir.vector<4 x !s32i>
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<4 x !s32i> -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_cmpneqps
  // LLVM: fcmp une <4 x float>

  // OGCG-LABEL: @test_cmpneqps
  // OGCG: fcmp une <4 x float>
  return __builtin_ia32_cmpneqps(a, b);
}

v4f test_rsqrtps(v4f a) {
  // CIR-LABEL: @test_rsqrtps
  // CIR: cir.call_llvm_intrinsic "x86.sse.rsqrt.ps" {{.*}} : (!cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_rsqrtps
  // LLVM: call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> {{.*}})

  // OGCG-LABEL: @test_rsqrtps
  // OGCG: call <4 x float> @llvm.x86.sse.rsqrt.ps(<4 x float> {{.*}})
  return __builtin_ia32_rsqrtps(a);
}

v4f test_rcpps(v4f a) {
  // CIR-LABEL: @test_rcpps
  // CIR: cir.call_llvm_intrinsic "x86.sse.rcp.ps" {{.*}} : (!cir.vector<4 x !cir.float>) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_rcpps
  // LLVM: call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> {{.*}})

  // OGCG-LABEL: @test_rcpps
  // OGCG: call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> {{.*}})
  return __builtin_ia32_rcpps(a);
}

v2d test_minpd(v2d a, v2d b) {
  // CIR-LABEL: @test_minpd
  // CIR: cir.call_llvm_intrinsic "x86.sse2.min.pd" {{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: @test_minpd
  // LLVM: call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> {{.*}}, <2 x double> {{.*}})

  // OGCG-LABEL: @test_minpd
  // OGCG: call <2 x double> @llvm.x86.sse2.min.pd(<2 x double> {{.*}}, <2 x double> {{.*}})
  return __builtin_ia32_minpd(a, b);
}

v32h test_addph512(v32h a, v32h b) {
  // CIR-LABEL: @test_addph512
  // CIR: cir.call_llvm_intrinsic "x86.avx512fp16.add.ph.512" {{.*}} : (!cir.vector<32 x !cir.f16>, !cir.vector<32 x !cir.f16>, !s32i) -> !cir.vector<32 x !cir.f16>

  // LLVM-LABEL: @test_addph512
  // LLVM: call <32 x half> @llvm.x86.avx512fp16.add.ph.512(<32 x half> {{.*}}, <32 x half> {{.*}}, i32 4)

  // OGCG-LABEL: @test_addph512
  // OGCG: call <32 x half> @llvm.x86.avx512fp16.add.ph.512(<32 x half> {{.*}}, <32 x half> {{.*}}, i32 4)
  return __builtin_ia32_addph512(a, b, 4);
}

v8bf test_cvtne2ps2bf16_128(v4f a, v4f b) {
  // CIR-LABEL: @test_cvtne2ps2bf16_128
  // CIR: cir.call_llvm_intrinsic "x86.avx512bf16.cvtne2ps2bf16.128" {{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>) -> !cir.vector<8 x !cir.bf16>

  // LLVM-LABEL: @test_cvtne2ps2bf16_128
  // LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> {{.*}}, <4 x float> {{.*}})

  // OGCG-LABEL: @test_cvtne2ps2bf16_128
  // OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.cvtne2ps2bf16.128(<4 x float> {{.*}}, <4 x float> {{.*}})
  return __builtin_ia32_cvtne2ps2bf16_128(a, b);
}

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
