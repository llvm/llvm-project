// RUN: %clang -target x86_64-unknown-fuchsia -fprofile-instr-generate -fcoverage-mapping -emit-llvm -S %s -o - | FileCheck %s

// CHECK-NOT: @__llvm_profile_runtime
static int f0() {
  return 100;
}
