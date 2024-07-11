// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=thunderx2t99 | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s

// CHECK: Extensions enabled for the given AArch64 target
// CHECK-EMPTY:
// CHECK-NEXT:     Architecture Feature(s)                                Description
// CHECK-NEXT:     FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-NEXT:     FEAT_CRC32                                             Enable ARMv8 CRC-32 checksum instructions
// CHECK-NEXT:     FEAT_FP                                                Enable ARMv8
// CHECK-NEXT:     FEAT_LOR                                               Enables ARM v8.1 Limited Ordering Regions extension
// CHECK-NEXT:     FEAT_LSE                                               Enable ARMv8.1 Large System Extension (LSE) atomic instructions
// CHECK-NEXT:     FEAT_PAN                                               Enables ARM v8.1 Privileged Access-Never extension
// CHECK-NEXT:     FEAT_RDM                                               Enable ARMv8.1 Rounding Double Multiply Add/Subtract instructions
// CHECK-NEXT:     FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support
// CHECK-NEXT:     FEAT_VHE                                               Enables ARM v8.1 Virtual Host extension
