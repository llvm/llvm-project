// RUN: inter-target-info bmg | FileCheck %s --check-prefix=INFO
// RUN: not inter-target-info xe3 2>&1 | FileCheck %s --check-prefix=CHIP
// RUN: not inter-target-info bmg future 2>&1 | FileCheck %s --check-prefix=FEATURE
// RUN: not inter-opt %s 2>&1 | FileCheck %s --check-prefix=ATTR

// INFO: chip: bmg
// INFO-NEXT: architecture: xe2
// INFO-NEXT: grf-byte-size: 64
// INFO-NEXT: grf-count: 128
// INFO-NEXT: simd-widths: 8 16 32
// INFO-NEXT: zebin-product-family: 1274
// INFO-NEXT: zebin-graphics-core: 3081
// INFO-NEXT: zebin-target-metadata: 0
// INFO-NEXT: zebin-product-config: 83902464
// INFO-NEXT: zebin-version: 1.64
// INFO-NEXT: first-explicit-argument: 24
// INFO-NEXT: cross-thread-payload-limit: 192
// INFO-NEXT: inline-payload-size: 32
// INFO-NEXT: payload-chunk-size: 64
// INFO-NEXT: reserved-payload-grfs: 5
// INFO-NEXT: address-space-0: private 64 64
// INFO-NEXT: address-space-1: global 64 64
// INFO-NEXT: address-space-2: constant 64 64
// INFO-NEXT: address-space-3: local 32 32
// INFO-NEXT: address-space-4: generic 64 64

// CHIP: unsupported Intel GPU target chip 'xe3'
// FEATURE: unsupported Intel GPU target feature 'future' for chip 'bmg'
// ATTR: unsupported Intel GPU target chip 'xe3'

func.func @invalid_target() attributes {
  xemachine.target = #xemachine.target<chip = "xe3">
}
