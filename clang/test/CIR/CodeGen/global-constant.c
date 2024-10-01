// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

const int global_no_use = 12;
// CIR: cir.global constant {{.*}}@global_no_use
// LLVM: @global_no_use = constant

const float global_used = 1.2f;
// CIR: cir.global constant {{.*}}@global_used
// LLVM: @global_used = constant

float const * get_float_ptr() {
  return &global_used;
}
