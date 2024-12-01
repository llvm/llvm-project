// REQUIRES: aarch64-registered-target
// RUN: %clang --target=aarch64-linux-gnu --print-supported-extensions | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s

// CHECK: All available -march extensions for AArch64
// CHECK-EMPTY:
// CHECK-NEXT:     Name                Architecture Feature(s)                                Description
// CHECK-NEXT:     aes                 FEAT_AES, FEAT_PMULL                                   Enable AES support
// CHECK-NEXT:     bf16                FEAT_BF16                                              Enable BFloat16 Extension
// CHECK-NEXT:     brbe                FEAT_BRBE                                              Enable Branch Record Buffer Extension
// CHECK-NEXT:     bti                 FEAT_BTI                                               Enable Branch Target Identification
// CHECK-NEXT:     cmpbr               FEAT_CMPBR                                             Enable Armv9.6-A base compare and branch instructions
// CHECK-NEXT:     fcma                FEAT_FCMA                                              Enable Armv8.3-A Floating-point complex number support
// CHECK-NEXT:     cpa                 FEAT_CPA                                               Enable Armv9.5-A Checked Pointer Arithmetic
// CHECK-NEXT:     crc                 FEAT_CRC32                                             Enable Armv8.0-A CRC-32 checksum instructions
// CHECK-NEXT:     crypto              FEAT_Crypto                                            Enable cryptographic instructions
// CHECK-NEXT:     cssc                FEAT_CSSC                                              Enable Common Short Sequence Compression (CSSC) instructions
// CHECK-NEXT:     d128                FEAT_D128, FEAT_LVA3, FEAT_SYSREG128, FEAT_SYSINSTR128 Enable Armv9.4-A 128-bit Page Table Descriptors, System Registers and instructions
// CHECK-NEXT:     dit                 FEAT_DIT                                               Enable Armv8.4-A Data Independent Timing instructions
// CHECK-NEXT:     dotprod             FEAT_DotProd                                           Enable dot product support
// CHECK-NEXT:     f32mm               FEAT_F32MM                                             Enable Matrix Multiply FP32 Extension
// CHECK-NEXT:     f64mm               FEAT_F64MM                                             Enable Matrix Multiply FP64 Extension
// CHECK-NEXT:     f8f16mm             FEAT_F8F16MM                                           Enable Armv9.6-A FP8 to Half-Precision Matrix Multiplication
// CHECK-NEXT:     f8f32mm             FEAT_F8F32MM                                           Enable Armv9.6-A FP8 to Single-Precision Matrix Multiplication
// CHECK-NEXT:     faminmax            FEAT_FAMINMAX                                          Enable FAMIN and FAMAX instructions
// CHECK-NEXT:     flagm               FEAT_FlagM                                             Enable Armv8.4-A Flag Manipulation instructions
// CHECK-NEXT:     fp                  FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// CHECK-NEXT:     fp16fml             FEAT_FHM                                               Enable FP16 FML instructions
// CHECK-NEXT:     fp8                 FEAT_FP8                                               Enable FP8 instructions
// CHECK-NEXT:     fp8dot2             FEAT_FP8DOT2                                           Enable FP8 2-way dot instructions
// CHECK-NEXT:     fp8dot4             FEAT_FP8DOT4                                           Enable FP8 4-way dot instructions
// CHECK-NEXT:     fp8fma              FEAT_FP8FMA                                            Enable Armv9.5-A FP8 multiply-add instructions
// CHECK-NEXT:     fprcvt              FEAT_FPRCVT                                            Enable Armv9.6-A base convert instructions for SIMD&FP scalar register operands of different input and output sizes
// CHECK-NEXT:     fp16                FEAT_FP16                                              Enable half-precision floating-point data processing
// CHECK-NEXT:     gcs                 FEAT_GCS                                               Enable Armv9.4-A Guarded Call Stack Extension
// CHECK-NEXT:     hbc                 FEAT_HBC                                               Enable Armv8.8-A Hinted Conditional Branches Extension
// CHECK-NEXT:     i8mm                FEAT_I8MM                                              Enable Matrix Multiply Int8 Extension
// CHECK-NEXT:     ite                 FEAT_ITE                                               Enable Armv9.4-A Instrumentation Extension
// CHECK-NEXT:     jscvt               FEAT_JSCVT                                             Enable Armv8.3-A JavaScript FP conversion instructions
// CHECK-NEXT:     ls64                FEAT_LS64, FEAT_LS64_V, FEAT_LS64_ACCDATA              Enable Armv8.7-A LD64B/ST64B Accelerator Extension
// CHECK-NEXT:     lse                 FEAT_LSE                                               Enable Armv8.1-A Large System Extension (LSE) atomic instructions
// CHECK-NEXT:     lse128              FEAT_LSE128                                            Enable Armv9.4-A 128-bit Atomic instructions
// CHECK-NEXT:     lsfe                FEAT_LSFE                                              Enable Armv9.6-A base Atomic floating-point in-memory instructions
// CHECK-NEXT:     lsui                FEAT_LSUI                                              Enable Armv9.6-A unprivileged load/store instructions
// CHECK-NEXT:     lut                 FEAT_LUT                                               Enable Lookup Table instructions
// CHECK-NEXT:     mops                FEAT_MOPS                                              Enable Armv8.8-A memcpy and memset acceleration instructions
// CHECK-NEXT:     memtag              FEAT_MTE, FEAT_MTE2                                    Enable Memory Tagging Extension
// CHECK-NEXT:     simd                FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// CHECK-NEXT:     occmo               FEAT_OCCMO                                             Enable Armv9.6-A Outer cacheable cache maintenance operations
// CHECK-NEXT:     pauth               FEAT_PAuth                                             Enable Armv8.3-A Pointer Authentication extension
// CHECK-NEXT:     pauth-lr            FEAT_PAuth_LR                                          Enable Armv9.5-A PAC enhancements
// CHECK-NEXT:     pcdphint            FEAT_PCDPHINT                                          Enable Armv9.6-A Producer Consumer Data Placement hints
// CHECK-NEXT:     pmuv3               FEAT_PMUv3                                             Enable Armv8.0-A PMUv3 Performance Monitors extension
// CHECK-NEXT:     pops                FEAT_PoPS                                              Enable Armv9.6-A Point Of Physical Storage (PoPS) DC instructions
// CHECK-NEXT:     predres             FEAT_SPECRES                                           Enable Armv8.5-A execution and data prediction invalidation instructions
// CHECK-NEXT:     rng                 FEAT_RNG                                               Enable Random Number generation instructions
// CHECK-NEXT:     ras                 FEAT_RAS, FEAT_RASv1p1                                 Enable Armv8.0-A Reliability, Availability and Serviceability Extensions
// CHECK-NEXT:     rasv2               FEAT_RASv2                                             Enable Armv8.9-A Reliability, Availability and Serviceability Extensions
// CHECK-NEXT:     rcpc                FEAT_LRCPC                                             Enable support for RCPC extension
// CHECK-NEXT:     rcpc3               FEAT_LRCPC3                                            Enable Armv8.9-A RCPC instructions for A64 and Advanced SIMD and floating-point instruction set
// CHECK-NEXT:     rdm                 FEAT_RDM                                               Enable Armv8.1-A Rounding Double Multiply Add/Subtract instructions
// CHECK-NEXT:     sb                  FEAT_SB                                                Enable Armv8.5-A Speculation Barrier
// CHECK-NEXT:     sha2                FEAT_SHA1, FEAT_SHA256                                 Enable SHA1 and SHA256 support
// CHECK-NEXT:     sha3                FEAT_SHA3, FEAT_SHA512                                 Enable SHA512 and SHA3 support
// CHECK-NEXT:     sm4                 FEAT_SM4, FEAT_SM3                                     Enable SM3 and SM4 support
// CHECK-NEXT:     sme                 FEAT_SME                                               Enable Scalable Matrix Extension (SME)
// CHECK-NEXT:     sme-b16b16          FEAT_SME_B16B16                                        Enable SME2.1 ZA-targeting non-widening BFloat16 instructions
// CHECK-NEXT:     sme-f16f16          FEAT_SME_F16F16                                        Enable SME non-widening Float16 instructions
// CHECK-NEXT:     sme-f64f64          FEAT_SME_F64F64                                        Enable Scalable Matrix Extension (SME) F64F64 instructions
// CHECK-NEXT:     sme-f8f16           FEAT_SME_F8F16                                         Enable Scalable Matrix Extension (SME) F8F16 instructions
// CHECK-NEXT:     sme-f8f32           FEAT_SME_F8F32                                         Enable Scalable Matrix Extension (SME) F8F32 instructions
// CHECK-NEXT:     sme-fa64            FEAT_SME_FA64                                          Enable the full A64 instruction set in streaming SVE mode
// CHECK-NEXT:     sme-i16i64          FEAT_SME_I16I64                                        Enable Scalable Matrix Extension (SME) I16I64 instructions
// CHECK-NEXT:     sme-lutv2           FEAT_SME_LUTv2                                         Enable Scalable Matrix Extension (SME) LUTv2 instructions
// CHECK-NEXT:     sme2                FEAT_SME2                                              Enable Scalable Matrix Extension 2 (SME2) instructions
// CHECK-NEXT:     sme2p1              FEAT_SME2p1                                            Enable Scalable Matrix Extension 2.1 instructions
// CHECK-NEXT:     sme2p2              FEAT_SME2p2                                            Enable Armv9.6-A Scalable Matrix Extension 2.2 instructions
// CHECK-NEXT:     profile             FEAT_SPE                                               Enable Statistical Profiling extension
// CHECK-NEXT:     predres2            FEAT_SPECRES2                                          Enable Speculation Restriction Instruction
// CHECK-NEXT:     ssbs                FEAT_SSBS, FEAT_SSBS2                                  Enable Speculative Store Bypass Safe bit
// CHECK-NEXT:     ssve-aes            FEAT_SSVE_AES                                          Enable Armv9.6-A SVE AES support in streaming SVE mode
// CHECK-NEXT:     ssve-fp8dot2        FEAT_SSVE_FP8DOT2                                      Enable SVE2 FP8 2-way dot product instructions
// CHECK-NEXT:     ssve-fp8dot4        FEAT_SSVE_FP8DOT4                                      Enable SVE2 FP8 4-way dot product instructions
// CHECK-NEXT:     ssve-fp8fma         FEAT_SSVE_FP8FMA                                       Enable SVE2 FP8 multiply-add instructions
// CHECK-NEXT:     sve                 FEAT_SVE                                               Enable Scalable Vector Extension (SVE) instructions
// CHECK-NEXT:     sve-aes             FEAT_SVE_AES, FEAT_SVE_PMULL128                        Enable SVE AES and quadword SVE polynomial multiply instructions
// CHECK-NEXT:     sve-aes2            FEAT_SVE_AES2                                          Enable Armv9.6-A SVE multi-vector AES and multi-vector quadword polynomial multiply instructions
// CHECK-NEXT:     sve-b16b16          FEAT_SVE_B16B16                                        Enable SVE2 non-widening and SME2 Z-targeting non-widening BFloat16 instructions
// CHECK-NEXT:     sve-bfscale         FEAT_SVE_BFSCALE                                       Enable Armv9.6-A SVE BFloat16 scaling instructions
// CHECK-NEXT:     sve-f16f32mm        FEAT_SVE_F16F32MM                                      Enable Armv9.6-A FP16 to FP32 Matrix Multiply
// CHECK-NEXT:     sve2                FEAT_SVE2                                              Enable Scalable Vector Extension 2 (SVE2) instructions
// CHECK-NEXT:     sve2-aes                                                                   Shorthand for +sve2+sve-aes
// CHECK-NEXT:     sve2-bitperm        FEAT_SVE_BitPerm                                       Enable bit permutation SVE2 instructions
// CHECK-NEXT:     sve2-sha3           FEAT_SVE_SHA3                                          Enable SHA3 SVE2 instructions
// CHECK-NEXT:     sve2-sm4            FEAT_SVE_SM4                                           Enable SM4 SVE2 instructions
// CHECK-NEXT:     sve2p1              FEAT_SVE2p1                                            Enable Scalable Vector Extension 2.1 instructions
// CHECK-NEXT:     sve2p2              FEAT_SVE2p2                                            Enable Armv9.6-A Scalable Vector Extension 2.2 instructions
// CHECK-NEXT:     the                 FEAT_THE                                               Enable Armv8.9-A Translation Hardening Extension
// CHECK-NEXT:     tlbiw               FEAT_TLBIW                                             Enable Armv9.5-A TLBI VMALL for Dirty State
// CHECK-NEXT:     tme                 FEAT_TME                                               Enable Transactional Memory Extension
// CHECK-NEXT:     wfxt                FEAT_WFxT                                              Enable Armv8.7-A WFET and WFIT instruction
