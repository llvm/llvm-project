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
