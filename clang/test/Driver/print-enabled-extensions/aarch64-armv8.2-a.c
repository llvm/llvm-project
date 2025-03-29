// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64 --print-enabled-extensions -march=armv8.2-a | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s

// CHECK: Extensions enabled for the given AArch64 target
// CHECK-EMPTY:
// CHECK-NEXT:     Architecture Feature(s)                                Description
// CHECK-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-NEXT:     FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-NEXT:     FEAT_DPB                                               Enable Armv8.2-A data Cache Clean to Point of Persistence
// CHECK-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// CHECK-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-NEXT:     FEAT_LOR                                               Enable Armv8.1-A Limited Ordering Regions extension
// CHECK-NEXT:     FEAT_LSE                                               Enable Armv8.1-A Large System Extension (LSE) atomic instructions
// CHECK-NEXT:     FEAT_PAN                                               Enable Armv8.1-A Privileged Access-Never extension
// CHECK-NEXT:     FEAT_PAN2                                              Enable Armv8.2-A PAN s1e1R and s1e1W Variants
// CHECK-NEXT:     FEAT_RAS, FEAT_RASv1p1                                 Enable Armv8.0-A Reliability, Availability and Serviceability Extensions
// CHECK-NEXT:     FEAT_RDM                                               Enable Armv8.1-A Rounding Double Multiply Add/Subtract instructions
// CHECK-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension
// CHECK-NEXT:     FEAT_UAO                                               Enable Armv8.2-A UAO PState
// CHECK-NEXT:     FEAT_VHE                                               Enable Armv8.1-A Virtual Host extension
