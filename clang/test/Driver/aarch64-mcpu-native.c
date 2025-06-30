// REQUIRES: aarch64-registered-target,system-linux,aarch64-host
// RUN: export LLVM_CPUINFO=%S/Inputs/cpunative/grace
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=native | FileCheck --strict-whitespace --check-prefix=CHECK-FEAT-GRACE --implicit-check-not=FEAT_ %s

// CHECK-FEAT-GRACE: Extensions enabled for the given AArch64 target
// CHECK-FEAT-GRACE-EMPTY:
// CHECK-FEAT-GRACE:     Architecture Feature(s)                                Description
// CHECK-FEAT-GRACE:     FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-FEAT-GRACE:     FEAT_AMUv1                                             Enable Armv8.4-A Activity Monitors extension
// CHECK-FEAT-GRACE:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-FEAT-GRACE:     FEAT_BF16                                              Enable BFloat16 Extension
// CHECK-FEAT-GRACE:     FEAT_BTI                                               Enable Branch Target Identification
// CHECK-FEAT-GRACE:     FEAT_CCIDX                                             Enable Armv8.3-A Extend of the CCSIDR number of sets
// CHECK-FEAT-GRACE:     FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-FEAT-GRACE:     FEAT_CSV2_2                                            Enable architectural speculation restriction
// CHECK-FEAT-GRACE:     FEAT_DIT                                               Enable Armv8.4-A Data Independent Timing instructions
// CHECK-FEAT-GRACE:     FEAT_DPB                                               Enable Armv8.2-A data Cache Clean to Point of Persistence
// CHECK-FEAT-GRACE:     FEAT_DPB2                                              Enable Armv8.5-A Cache Clean to Point of Deep Persistence
// CHECK-FEAT-GRACE:     FEAT_DotProd                                           Enable dot product support
// CHECK-FEAT-GRACE:     FEAT_ETE                                               Enable Embedded Trace Extension
// CHECK-FEAT-GRACE:     FEAT_FCMA                                              Enable Armv8.3-A Floating-point complex number support
// CHECK-FEAT-GRACE:     FEAT_FHM                                               Enable FP16 FML instructions
// CHECK-FEAT-GRACE:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-FEAT-GRACE:     FEAT_FP16                                              Enable half-precision floating-point data processing
// CHECK-FEAT-GRACE:     FEAT_FPAC                                              Enable Armv8.3-A Pointer Authentication Faulting enhancement
// CHECK-FEAT-GRACE:     FEAT_FRINTTS                                           Enable FRInt[32|64][Z|X] instructions that round a floating-point number to an integer (in FP format) forcing it to fit into a 32- or 64-bit int
// CHECK-FEAT-GRACE:     FEAT_FlagM                                             Enable Armv8.4-A Flag Manipulation instructions
// CHECK-FEAT-GRACE:     FEAT_FlagM2                                            Enable alternative NZCV format for floating point comparisons
// CHECK-FEAT-GRACE:     FEAT_I8MM                                              Enable Matrix Multiply Int8 Extension
// CHECK-FEAT-GRACE:     FEAT_JSCVT                                             Enable Armv8.3-A JavaScript FP conversion instructions
// CHECK-FEAT-GRACE:     FEAT_LOR                                               Enable Armv8.1-A Limited Ordering Regions extension
// CHECK-FEAT-GRACE:     FEAT_LRCPC                                             Enable support for RCPC extension
// CHECK-FEAT-GRACE:     FEAT_LRCPC2                                            Enable Armv8.4-A RCPC instructions with Immediate Offsets
// CHECK-FEAT-GRACE:     FEAT_LSE                                               Enable Armv8.1-A Large System Extension (LSE) atomic instructions
// CHECK-FEAT-GRACE:     FEAT_LSE2                                              Enable Armv8.4-A Large System Extension 2 (LSE2) atomicity rules
// CHECK-FEAT-GRACE:     FEAT_MPAM                                              Enable Armv8.4-A Memory system Partitioning and Monitoring extension
// CHECK-FEAT-GRACE:     FEAT_MTE, FEAT_MTE2                                    Enable Memory Tagging Extension
// CHECK-FEAT-GRACE:     FEAT_NV, FEAT_NV2                                      Enable Armv8.4-A Nested Virtualization Enchancement
// CHECK-FEAT-GRACE:     FEAT_PAN                                               Enable Armv8.1-A Privileged Access-Never extension
// CHECK-FEAT-GRACE:     FEAT_PAN2                                              Enable Armv8.2-A PAN s1e1R and s1e1W Variants
// CHECK-FEAT-GRACE:     FEAT_PAuth                                             Enable Armv8.3-A Pointer Authentication extension
// CHECK-FEAT-GRACE:     FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension
// CHECK-FEAT-GRACE:     FEAT_RAS, FEAT_RASv1p1                                 Enable Armv8.0-A Reliability, Availability and Serviceability Extensions
// CHECK-FEAT-GRACE:     FEAT_RDM                                               Enable Armv8.1-A Rounding Double Multiply Add/Subtract instructions
// CHECK-FEAT-GRACE:     FEAT_RNG                                               Enable Random Number generation instructions
// CHECK-FEAT-GRACE:     FEAT_SB                                                Enable Armv8.5-A Speculation Barrier
// CHECK-FEAT-GRACE:     FEAT_SEL2                                              Enable Armv8.4-A Secure Exception Level 2 extension
// CHECK-FEAT-GRACE:     FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support
// CHECK-FEAT-GRACE:     FEAT_SHA3, FEAT_SHA512                                 Enable SHA512 and SHA3 support
// CHECK-FEAT-GRACE:     FEAT_SM4, FEAT_SM3                                     Enable SM3 and SM4 support
// CHECK-FEAT-GRACE:     FEAT_SPE                                               Enable Statistical Profiling extension
// CHECK-FEAT-GRACE:     FEAT_SPECRES                                           Enable Armv8.5-A execution and data prediction invalidation instructions
// CHECK-FEAT-GRACE:     FEAT_SSBS, FEAT_SSBS2                                  Enable Speculative Store Bypass Safe bit
// CHECK-FEAT-GRACE:     FEAT_SVE                                               Enable Scalable Vector Extension (SVE) instructions
// CHECK-FEAT-GRACE:     FEAT_SVE2                                              Enable Scalable Vector Extension 2 (SVE2) instructions
// CHECK-FEAT-GRACE:     FEAT_SVE_AES, FEAT_SVE_PMULL128                        Enable SVE AES and quadword SVE polynomial multiply instructions
// CHECK-FEAT-GRACE:     FEAT_SVE_BitPerm                                       Enable bit permutation SVE2 instructions
// CHECK-FEAT-GRACE:     FEAT_SVE_SHA3                                          Enable SVE SHA3 instructions
// CHECK-FEAT-GRACE:     FEAT_SVE_SM4                                           Enable SM4 SVE2 instructions
// CHECK-FEAT-GRACE:     FEAT_TLBIOS, FEAT_TLBIRANGE                            Enable Armv8.4-A TLB Range and Maintenance instructions
// CHECK-FEAT-GRACE:     FEAT_TRBE                                              Enable Trace Buffer Extension
// CHECK-FEAT-GRACE:     FEAT_TRF                                               Enable Armv8.4-A Trace extension
// CHECK-FEAT-GRACE:     FEAT_UAO                                               Enable Armv8.2-A UAO PState
// CHECK-FEAT-GRACE:     FEAT_VHE                                               Enable Armv8.1-A Virtual Host extension

// RUN: export LLVM_CPUINFO=%S/Inputs/cpunative/neoverse-n1
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=native | FileCheck --strict-whitespace --check-prefix=CHECK-FEAT-NN1 --implicit-check-not=FEAT_ %s

// CHECK-FEAT-NN1: Extensions enabled for the given AArch64 target
// CHECK-FEAT-NN1-EMPTY:
// CHECK-FEAT-NN1:    Architecture Feature(s)                                Description
// CHECK-FEAT-NN1:    FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-FEAT-NN1:    FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-FEAT-NN1:    FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-FEAT-NN1:    FEAT_DPB                                               Enable Armv8.2-A data Cache Clean to Point of Persistence
// CHECK-FEAT-NN1:    FEAT_DotProd                                           Enable dot product support
// CHECK-FEAT-NN1:    FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-FEAT-NN1:    FEAT_FP16                                              Enable half-precision floating-point data processing
// CHECK-FEAT-NN1:    FEAT_LOR                                               Enable Armv8.1-A Limited Ordering Regions extension
// CHECK-FEAT-NN1:    FEAT_LRCPC                                             Enable support for RCPC extension
// CHECK-FEAT-NN1:    FEAT_LSE                                               Enable Armv8.1-A Large System Extension (LSE) atomic instructions
// CHECK-FEAT-NN1:    FEAT_PAN                                               Enable Armv8.1-A Privileged Access-Never extension
// CHECK-FEAT-NN1:    FEAT_PAN2                                              Enable Armv8.2-A PAN s1e1R and s1e1W Variants
// CHECK-FEAT-NN1:    FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension
// CHECK-FEAT-NN1:    FEAT_RAS, FEAT_RASv1p1                                 Enable Armv8.0-A Reliability, Availability and Serviceability Extensions
// CHECK-FEAT-NN1:    FEAT_RDM                                               Enable Armv8.1-A Rounding Double Multiply Add/Subtract instructions
// CHECK-FEAT-NN1:    FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support
// CHECK-FEAT-NN1:    FEAT_SPE                                               Enable Statistical Profiling extension
// CHECK-FEAT-NN1:    FEAT_SSBS, FEAT_SSBS2                                  Enable Speculative Store Bypass Safe bit
// CHECK-FEAT-NN1:    FEAT_UAO                                               Enable Armv8.2-A UAO PState
// CHECK-FEAT-NN1:    FEAT_VHE                                               Enable Armv8.1-A Virtual Host extension


// RUN: export LLVM_CPUINFO=%S/Inputs/cpunative/cortex-a57
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=native | FileCheck --strict-whitespace --check-prefix=CHECK-FEAT-CA57 --implicit-check-not=FEAT_ %s

// CHECK-FEAT-CA57: Extensions enabled for the given AArch64 target
// CHECK-FEAT-CA57-EMPTY:
// CHECK-FEAT-CA57:    Architecture Feature(s)                                Description
// CHECK-FEAT-CA57:    FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-FEAT-CA57:    FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-FEAT-CA57:    FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-FEAT-CA57:    FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-FEAT-CA57:    FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension
// CHECK-FEAT-CA57:    FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support

// RUN: export LLVM_CPUINFO=%S/Inputs/cpunative/cortex-a72
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=native | FileCheck --strict-whitespace  --check-prefix=CHECK-FEAT-CA72 --implicit-check-not=FEAT_ %s

// CHECK-FEAT-CA72: Extensions enabled for the given AArch64 target
// CHECK-EMPTY:
// CHECK-FEAT-CA72:   Architecture Feature(s)                                Description
// CHECK-FEAT-CA72:    FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-FEAT-CA72:    FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-FEAT-CA72:    FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-FEAT-CA72:    FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-FEAT-CA72:    FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension
// CHECK-FEAT-CA72:    FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support

// RUN: export LLVM_CPUINFO=%S/Inputs/cpunative/cortex-a76
// RUN: %clang --target=aarch64 --print-enabled-extensions -mcpu=native | FileCheck --strict-whitespace --check-prefix=CHECK-FEAT-CA76 --implicit-check-not=FEAT_ %s

// CHECK-FEAT-CA76: Extensions enabled for the given AArch64 target
// CHECK-FEAT-CA76-EMPTY:
// CHECK-FEAT-CA76:    Architecture Feature(s)                                Description
// CHECK-FEAT-CA76:    FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-FEAT-CA76:    FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-FEAT-CA76:    FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-FEAT-CA76:    FEAT_DPB                                               Enable Armv8.2-A data Cache Clean to Point of Persistence
// CHECK-FEAT-CA76:    FEAT_DotProd                                           Enable dot product support
// CHECK-FEAT-CA76:    FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-FEAT-CA76:    FEAT_FP16                                              Enable half-precision floating-point data processing
// CHECK-FEAT-CA76:    FEAT_LOR                                               Enable Armv8.1-A Limited Ordering Regions extension
// CHECK-FEAT-CA76:    FEAT_LRCPC                                             Enable support for RCPC extension
// CHECK-FEAT-CA76:    FEAT_LSE                                               Enable Armv8.1-A Large System Extension (LSE) atomic instructions
// CHECK-FEAT-CA76:    FEAT_PAN                                               Enable Armv8.1-A Privileged Access-Never extension
// CHECK-FEAT-CA76:    FEAT_PAN2                                              Enable Armv8.2-A PAN s1e1R and s1e1W Variants
// CHECK-FEAT-CA76:    FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension
// CHECK-FEAT-CA76:    FEAT_RAS, FEAT_RASv1p1                                 Enable Armv8.0-A Reliability, Availability and Serviceability Extensions
// CHECK-FEAT-CA76:    FEAT_RDM                                               Enable Armv8.1-A Rounding Double Multiply Add/Subtract instructions
// CHECK-FEAT-CA76:    FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support
// CHECK-FEAT-CA76:    FEAT_SSBS, FEAT_SSBS2                                  Enable Speculative Store Bypass Safe bit
// CHECK-FEAT-CA76:    FEAT_UAO                                               Enable Armv8.2-A UAO PState
// CHECK-FEAT-CA76:    FEAT_VHE                                               Enable Armv8.1-A Virtual Host extension
