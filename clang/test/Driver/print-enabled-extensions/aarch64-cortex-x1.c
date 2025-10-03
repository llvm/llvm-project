// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=cortex-x1 | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s

// CHECK: Extensions enabled for the given AArch64 target
// CHECK-EMPTY:
// CHECK-NEXT:     Architecture Feature(s)                                Description
// CHECK-NEXT:     FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-NEXT:     FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-NEXT:     FEAT_DPB                                               Enable Armv8.2-A data Cache Clean to Point of Persistence
// CHECK-NEXT:     FEAT_DotProd                                           Enable dot product support
// CHECK-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-NEXT:     FEAT_FP16                                              Enable half-precision floating-point data processing
// CHECK-NEXT:     FEAT_LOR                                               Enable Armv8.1-A Limited Ordering Regions extension
// CHECK-NEXT:     FEAT_LRCPC                                             Enable support for RCPC extension
// CHECK-NEXT:     FEAT_LSE                                               Enable Armv8.1-A Large System Extension (LSE) atomic instructions
// CHECK-NEXT:     FEAT_PAN                                               Enable Armv8.1-A Privileged Access-Never extension
// CHECK-NEXT:     FEAT_PAN2                                              Enable Armv8.2-A PAN s1e1R and s1e1W Variants
// CHECK-NEXT:     FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension
// CHECK-NEXT:     FEAT_RAS, FEAT_RASv1p1                                 Enable Armv8.0-A Reliability, Availability and Serviceability Extensions
// CHECK-NEXT:     FEAT_RDM                                               Enable Armv8.1-A Rounding Double Multiply Add/Subtract instructions
// CHECK-NEXT:     FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support
// CHECK-NEXT:     FEAT_SPE                                               Enable Statistical Profiling extension
// CHECK-NEXT:     FEAT_SSBS, FEAT_SSBS2                                  Enable Speculative Store Bypass Safe bit
// CHECK-NEXT:     FEAT_UAO                                               Enable Armv8.2-A UAO PState
// CHECK-NEXT:     FEAT_VHE                                               Enable Armv8.1-A Virtual Host extension
