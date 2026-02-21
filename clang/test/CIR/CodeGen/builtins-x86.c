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
