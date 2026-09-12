// Verify that the llvm-xray tool can read XRay Mach-O instrumentation maps.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: %llvm_xray extract %t | FileCheck %s

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

// CHECK: ---
// CHECK: - { id: {{[0-9]+}}, address: 0x{{[0-9a-fA-F]+}}

[[clang::xray_always_instrument]] int fn_a() { return 1; }
[[clang::xray_always_instrument]] int fn_b() { return 2; }

int main() { return fn_a() + fn_b(); }
