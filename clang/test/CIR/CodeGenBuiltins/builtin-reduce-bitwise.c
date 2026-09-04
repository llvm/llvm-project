// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef int v4si __attribute__((vector_size(16)));
typedef unsigned int v4su __attribute__((vector_size(16)));

int test_reduce_or(v4si x) {
  // CIR-LABEL: @test_reduce_or
  // CIR: cir.call_llvm_intrinsic "vector.reduce.or"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_or
  // LLVM: call i32 @llvm.vector.reduce.or.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_or(x);
}

int test_reduce_and(v4si x) {
  // CIR-LABEL: @test_reduce_and
  // CIR: cir.call_llvm_intrinsic "vector.reduce.and"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_and
  // LLVM: call i32 @llvm.vector.reduce.and.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_and(x);
}

int test_reduce_xor(v4si x) {
  // CIR-LABEL: @test_reduce_xor
  // CIR: cir.call_llvm_intrinsic "vector.reduce.xor"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_xor
  // LLVM: call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_xor(x);
}

unsigned test_reduce_or_unsigned(v4su x) {
  // CIR-LABEL: @test_reduce_or_unsigned
  // CIR: cir.call_llvm_intrinsic "vector.reduce.or"
  // CIR: cir.return
  // LLVM-LABEL: @test_reduce_or_unsigned
  // LLVM: call i32 @llvm.vector.reduce.or.v4i32(<4 x i32>
  // LLVM: ret i32
  return __builtin_reduce_or(x);
}
