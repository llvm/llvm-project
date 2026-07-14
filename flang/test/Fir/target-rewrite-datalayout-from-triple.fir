// RUN: fir-opt --target-rewrite="target=s390x-unknown-linux-gnu" %s | FileCheck %s

// Test that target-rewrite derives correct data layout from target triple
// when no llvm.data_layout is present. For SystemZ, f128 ABI alignment
// must be 8 bytes (64 bits), not 16.

module attributes {
  llvm.target_triple = "s390x-unknown-linux-gnu"
} {
}

// CHECK: dlti.dl_spec
// CHECK: f128 = dense<64> : vector<2xi64>
