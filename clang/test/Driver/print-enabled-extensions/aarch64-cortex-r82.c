// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=cortex-r82 | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s

// CHECK: Extensions enabled for the given AArch64 target
// CHECK-EMPTY:
// CHECK-NEXT:     Architecture Feature(s)                                Description
// CHECK-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-NEXT:     FEAT_CCIDX                                             Enable v8.3-A Extend of the CCSIDR number of sets
// CHECK-NEXT:     FEAT_CRC32                                             Enable ARMv8 CRC-32 checksum instructions
// CHECK-NEXT:     FEAT_CSV2_2                                            Enable architectural speculation restriction
// CHECK-NEXT:     FEAT_DIT                                               Enable v8.4-A Data Independent Timing instructions
// CHECK-NEXT:     FEAT_DPB                                               Enable v8.2 data Cache Clean to Point of Persistence
// CHECK-NEXT:     FEAT_DPB2                                              Enable v8.5 Cache Clean to Point of Deep Persistence
// CHECK-NEXT:     FEAT_DotProd                                           Enable dot product support
// CHECK-NEXT:     FEAT_FCMA                                              Enable v8.3-A Floating-point complex number support
// CHECK-NEXT:     FEAT_FHM                                               Enable FP16 FML instructions
// CHECK-NEXT:     FEAT_FP                                                Enable ARMv8
// CHECK-NEXT:     FEAT_FP16                                              Full FP16
// CHECK-NEXT:     FEAT_FlagM                                             Enable v8.4-A Flag Manipulation Instructions
// CHECK-NEXT:     FEAT_JSCVT                                             Enable v8.3-A JavaScript FP conversion instructions
// CHECK-NEXT:     FEAT_LRCPC                                             Enable support for RCPC extension
// CHECK-NEXT:     FEAT_LRCPC2                                            Enable v8.4-A RCPC instructions with Immediate Offsets
// CHECK-NEXT:     FEAT_LSE                                               Enable ARMv8.1 Large System Extension (LSE) atomic instructions
// CHECK-NEXT:     FEAT_PAN                                               Enables ARM v8.1 Privileged Access-Never extension
// CHECK-NEXT:     FEAT_PAN2                                              Enable v8.2 PAN s1e1R and s1e1W Variants
// CHECK-NEXT:     FEAT_PAuth                                             Enable v8.3-A Pointer Authentication extension
// CHECK-NEXT:     FEAT_PMUv3                                             Enable Code Generation for ARMv8 PMUv3 Performance Monitors extension
// CHECK-NEXT:     FEAT_RAS, FEAT_RASv1p1                                 Enable ARMv8 Reliability, Availability and Serviceability Extensions
// CHECK-NEXT:     FEAT_RDM                                               Enable ARMv8.1 Rounding Double Multiply Add/Subtract instructions
// CHECK-NEXT:     FEAT_SB                                                Enable v8.5 Speculation Barrier
// CHECK-NEXT:     FEAT_SEL2                                              Enable v8.4-A Secure Exception Level 2 extension
// CHECK-NEXT:     FEAT_SPECRES                                           Enable v8.5a execution and data prediction invalidation instructions
// CHECK-NEXT:     FEAT_SSBS, FEAT_SSBS2                                  Enable Speculative Store Bypass Safe bit
// CHECK-NEXT:     FEAT_TLBIOS, FEAT_TLBIRANGE                            Enable v8.4-A TLB Range and Maintenance Instructions
// CHECK-NEXT:     FEAT_TRF                                               Enable v8.4-A Trace extension
// CHECK-NEXT:     FEAT_UAO                                               Enable v8.2 UAO PState
