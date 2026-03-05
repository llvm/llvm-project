// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fno-PIE -S -Xclang -emit-cir %s -o %t1.cir
// RUN: FileCheck --input-file=%t1.cir %s -check-prefix=CIR
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -fno-PIE -S -Xclang -emit-llvm %s -o %t1.ll
// RUN: FileCheck --input-file=%t1.ll %s -check-prefix=LLVM

extern int var;
int get() {
  return var;
}
// CIR: cir.global "private" external dso_local @var : !s32i {alignment = 4 : i64}
// LLVM: @var = external dso_local global i32
