// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -menable-experimental-extensions \
// RUN: -march=rv32i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv32 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv32 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -menable-experimental-extensions \
// RUN: -march=rv64i_zicfilp1p0 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=unlabeled -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,UNLABELED-FLAG %s

// RUN: %clang --target=riscv64 -fcf-protection=branch \
// RUN: -mcf-branch-label-scheme=func-sig -S -emit-llvm %s -o - | FileCheck \
// RUN: --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=unlabeled -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,UNLABELED-SCHEME-UNUSED %s

// RUN: %clang --target=riscv64 -mcf-branch-label-scheme=func-sig -S \
// RUN: -emit-llvm %s -o - 2>&1 | FileCheck \
// RUN: --check-prefixes=NO-FLAG,FUNC-SIG-SCHEME-UNUSED %s

// Default -mcf-branch-label-scheme is func-sig
// RUN: %clang --target=riscv32 -fcf-protection=branch -S -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// Default -mcf-branch-label-scheme is func-sig
// RUN: %clang --target=riscv64 -fcf-protection=branch -S -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefixes=BRANCH-PROT-FLAG,FUNC-SIG-FLAG %s

// UNLABELED-SCHEME-UNUSED: warning: argument unused during compilation:
// UNLABELED-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=unlabeled'
// FUNC-SIG-SCHEME-UNUSED: warning: argument unused during compilation:
// FUNC-SIG-SCHEME-UNUSED-SAME: '-mcf-branch-label-scheme=func-sig'

// BRANCH-PROT-FLAG-DAG: [[P_FLAG:![0-9]+]] = !{i32 8, !"cf-protection-branch", i32 1}
// UNLABELED-FLAG-DAG: [[S_FLAG:![0-9]+]] = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
// FUNC-SIG-FLAG-DAG: [[S_FLAG:![0-9]+]] = !{i32 1, !"cf-branch-label-scheme", !"func-sig"}
// BRANCH-PROT-FLAG-DAG: !llvm.module.flags = !{{[{].*}}[[P_FLAG]]{{.*, }}[[S_FLAG]]{{(,.+)?[}]}}
// NO-FLAG-NOT: !{i32 8, !"cf-protection-branch", i32 1}
// NO-FLAG-NOT: !{i32 8, !"cf-branch-label-scheme", !"unlabeled"}
// NO-FLAG-NOT: !{i32 8, !"cf-branch-label-scheme", !"func-sig"}

int main() { return 0; }
