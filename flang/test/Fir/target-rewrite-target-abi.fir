// RUN: fir-opt --target-rewrite %s | FileCheck %s --check-prefixes=DEFAULT
// RUN: fir-opt --target-rewrite="target-abi=lp64d" %s | FileCheck %s --check-prefixes=LP64D
// RUN: fir-opt --target-rewrite="target-abi=lp64f" %s | FileCheck %s --check-prefixes=LP64F
// RUN: fir-opt --target-rewrite="target=riscv64-unknown-linux-gnu target-abi=lp64" %s | FileCheck %s --check-prefixes=LINUX

// Currently only RISCV64 uses the target-abi option.

// DEFAULT: module attributes {
// DEFAULT-SAME: fir.target_abi = "lp64"

// LP64D: module attributes {
// LP64D-SAME: fir.target_abi = "lp64d"

// LP64F: module attributes {
// LP64F-SAME: fir.target_abi = "lp64f"

// LINUX: module attributes {
// LINUX-SAME:  fir.target_abi = "lp64"

module attributes {fir.target_abi = "lp64"}  {
  func.func @dummyfunc() -> () {
    return
  }
}
