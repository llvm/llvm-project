// RUN: %clang -target riscv32-unknown-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix=RV32

// RV32: "target-features"="+a,+c,+m,+relax,-save-restore"
// RV64: "target-features"="+64bit,+a,+c,+m,+relax,-save-restore"

// Dummy function
int foo(void){
  return  3;
}
