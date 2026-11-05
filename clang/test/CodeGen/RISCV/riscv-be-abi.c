// RUN: %clang_cc1 -triple riscv32be -target-abi ilp32 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=RV32BE
// RUN: %clang_cc1 -triple riscv32 -target-abi ilp32 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=RV32LE
// RUN: %clang_cc1 -triple riscv64be -target-abi lp64 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=RV64BE
// RUN: %clang_cc1 -triple riscv64 -target-abi lp64 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=RV64LE

typedef unsigned long long u64;

struct aligned_2xlen_rv32 {
  u64 value;
};

// Preserve aggregate identity on big-endian targets. A bare i64 would be
// indistinguishable from a 2*XLen scalar in the backend, even though aggregates
// use memory-layout ordering while named scalars use significance ordering.

// RV32BE-LABEL: define dso_local [1 x i64] @pass_rv32(
// RV32BE-SAME: [1 x i64] [[X:%.*]])
// RV32LE-LABEL: define dso_local i64 @pass_rv32(
// RV32LE-SAME: i64 [[X:%.*]])
struct aligned_2xlen_rv32 pass_rv32(struct aligned_2xlen_rv32 x) {
  return x;
}

#if __riscv_xlen == 64
typedef unsigned __int128 u128;

struct aligned_2xlen_rv64 {
  u128 value;
};

// RV64BE-LABEL: define dso_local [1 x i128] @pass_rv64(
// RV64BE-SAME: [1 x i128] [[X:%.*]])
// RV64LE-LABEL: define dso_local i128 @pass_rv64(
// RV64LE-SAME: i128 [[X:%.*]])
struct aligned_2xlen_rv64 pass_rv64(struct aligned_2xlen_rv64 x) {
  return x;
}
#endif
