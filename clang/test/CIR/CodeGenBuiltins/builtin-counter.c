// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

// CIR-LABEL: cir.func {{.*}} @read_cycles(
// CIR: %{{.+}} = cir.call_llvm_intrinsic "readcyclecounter" : () -> !u64i
// LLVM-LABEL: @read_cycles(
// LLVM: %{{.+}} = call i64 @llvm.readcyclecounter()
unsigned long long read_cycles(void) {
  return __builtin_readcyclecounter();
}

// CIR-LABEL: cir.func {{.*}} @read_steady(
// CIR: %{{.+}} = cir.call_llvm_intrinsic "readsteadycounter" : () -> !u64i
// LLVM-LABEL: @read_steady(
// LLVM: %{{.+}} = call i64 @llvm.readsteadycounter()
unsigned long long read_steady(void) {
  return __builtin_readsteadycounter();
}
