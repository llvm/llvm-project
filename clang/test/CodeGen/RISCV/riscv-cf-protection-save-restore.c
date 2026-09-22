// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicsr_zimop_zicfiss -fcf-protection=return \
// RUN:   -msave-restore -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicsr_zimop_zicfiss -fcf-protection=full \
// RUN:   -msave-restore -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN:   -march=rv32i_zicsr_zimop_zicfiss -fcf-protection \
// RUN:   -msave-restore -S -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32 -march=rv32i_zicsr_zimop_zicfiss \
// RUN:   -fsanitize=shadow-call-stack -msave-restore -S \
// RUN:   -emit-llvm %s -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-SW %s

// CHECK: warning: option -fcf-protection=return is not supported with feature -msave-restore on RISC-V targets [-Winvalid-command-line-argument]
// CHECK-SW: warning: option -fsanitize=shadow-call-stack is not supported with feature -msave-restore on RISC-V targets [-Winvalid-command-line-argument]
int main() { return 0; }
