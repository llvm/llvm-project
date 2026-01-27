// REQUIRES: aarch64-registered-target,system-linux,aarch64-host
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
// CHECK-FEAT-CA57:    FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-FEAT-CA57:    FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-FEAT-CA57:    FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-FEAT-CA57:    FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension

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
