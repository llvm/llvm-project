// RUN: %clang -target armv8a-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8A %s
// CHECK-V8A: #define __ARMEL__ 1
// CHECK-V8A: #define __ARM_ARCH 8
// CHECK-V8A: #define __ARM_ARCH_8A__ 1
// CHECK-V8A: #define __ARM_FEATURE_CRC32 1
// CHECK-V8A: #define __ARM_FEATURE_DIRECTED_ROUNDING 1
// CHECK-V8A: #define __ARM_FEATURE_NUMERIC_MAXMIN 1
// CHECK-V8A-NOT: #define __ARM_FP 0x
// CHECK-V8A-NOT: #define __ARM_FEATURE_DOTPROD
// CHECK-V8A-NOT: #define __ARM_BF16_FORMAT_ALTERNATIVE
// CHECK-V8A-NOT: #define __ARM_FEATURE_BF16
// CHECK-V8A-NOT: #define __ARM_FEATURE_BF16_VECTOR_ARITHMETIC

// RUN: %clang -target armv8a-none-linux-gnueabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8A-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8a-none-linux-gnueabihf -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8A-ALLOW-FP-INSTR %s
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARMEL__ 1
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_ARCH 8
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_ARCH_8A__ 1
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_FEATURE_CRC32 1
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_FEATURE_DIRECTED_ROUNDING 1
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_FEATURE_NUMERIC_MAXMIN 1
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_FP 0xe
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_FP16_ARGS 1
// CHECK-V8A-ALLOW-FP-INSTR: #define __ARM_FP16_FORMAT_IEEE 1
// CHECK-V8A-ALLOW-FP-INSTR-V8A-NOT: #define __ARM_FEATURE_DOTPROD

// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2-a+nofp16fml+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2-a+nofp16+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2-a+fp16+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8-a+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8-a+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+nofp16fml+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+nofp16+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+fp16+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FP 0xe
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FP16_FORMAT_IEEE 1

// +fp16fml without neon doesn't make sense as the fp16fml instructions all require SIMD.
// However, as +fp16fml implies +fp16 there is a set of defines that we would expect.
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8-a+fp16fml -mfpu=vfp4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8-a+fp16 -mfpu=vfp4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+fp16fml -mfpu=vfp4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+fp16 -mfpu=vfp4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// CHECK-FULLFP16-SCALAR:       #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-SCALAR-NOT:   #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-SCALAR:       #define __ARM_FP 0xe
// CHECK-FULLFP16-SCALAR:       #define __ARM_FP16_FORMAT_IEEE 1

// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2-a -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2-a+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2-a+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2-a+fp16fml+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.4-a+fp16fml+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR-NOT: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR-NOT: #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR: #define __ARM_FP 0xe
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR: #define __ARM_FP16_FORMAT_IEEE 1

// RUN: %clang -target arm -march=armv8-a+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// RUN: %clang -target arm -march=armv8-a+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// RUN: %clang -target arm -march=armv8-a+fp16+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// RUN: %clang -target arm -march=armv8-a+fp16fml+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// RUN: %clang -target arm -march=armv8.4-a+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// RUN: %clang -target arm -march=armv8.4-a+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// RUN: %clang -target arm -march=armv8.4-a+fp16+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// RUN: %clang -target arm -march=armv8.4-a+fp16fml+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SOFT %s
// CHECK-FULLFP16-SOFT-NOT: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
// CHECK-FULLFP16-SOFT-NOT: #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
// CHECK-FULLFP16-SOFT-NOT: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
// CHECK-FULLFP16-SOFT-NOT: #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

// RUN: %clang -target arm-none-linux-gnueabi -march=armv8.2a+dotprod -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-DOTPROD %s
// CHECK-DOTPROD: #define __ARM_FEATURE_DOTPROD 1

// RUN: %clang -target armv8r-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8R %s
// CHECK-V8R: #define __ARMEL__ 1
// CHECK-V8R: #define __ARM_ARCH 8
// CHECK-V8R: #define __ARM_ARCH_8R__ 1
// CHECK-V8R: #define __ARM_FEATURE_CRC32 1
// CHECK-V8R: #define __ARM_FEATURE_DIRECTED_ROUNDING 1
// CHECK-V8R: #define __ARM_FEATURE_NUMERIC_MAXMIN 1
// CHECK-V8R-NOT: #define __ARM_FP 0x

// RUN: %clang -target armv8r-none-linux-gnueabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8R-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8r-none-linux-gnueabihf -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8R-ALLOW-FP-INSTR %s
// CHECK-V8R-ALLOW-FP-INSTR: #define __ARMEL__ 1
// CHECK-V8R-ALLOW-FP-INSTR: #define __ARM_ARCH 8
// CHECK-V8R-ALLOW-FP-INSTR: #define __ARM_ARCH_8R__ 1
// CHECK-V8R-ALLOW-FP-INSTR: #define __ARM_FEATURE_CRC32 1
// CHECK-V8R-ALLOW-FP-INSTR: #define __ARM_FEATURE_DIRECTED_ROUNDING 1
// CHECK-V8R-ALLOW-FP-INSTR: #define __ARM_FEATURE_NUMERIC_MAXMIN 1
// CHECK-V8R-ALLOW-FP-INSTR: #define __ARM_FP 0xe

// RUN: %clang -target armv7a-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V7 %s
// CHECK-V7: #define __ARMEL__ 1
// CHECK-V7: #define __ARM_ARCH 7
// CHECK-V7: #define __ARM_ARCH_7A__ 1
// CHECK-V7-NOT: __ARM_FEATURE_CRC32
// CHECK-V7-NOT: __ARM_FEATURE_NUMERIC_MAXMIN
// CHECK-V7-NOT: __ARM_FEATURE_DIRECTED_ROUNDING
// CHECK-V7-NOT: #define __ARM_FP 0x

// RUN: %clang -target armv7a-none-linux-gnueabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V7-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7a-none-linux-gnueabihf -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V7-ALLOW-FP-INSTR %s
// CHECK-V7-ALLOW-FP-INSTR: #define __ARMEL__ 1
// CHECK-V7-ALLOW-FP-INSTR: #define __ARM_ARCH 7
// CHECK-V7-ALLOW-FP-INSTR: #define __ARM_ARCH_7A__ 1
// CHECK-V7-ALLOW-FP-INSTR-NOT: __ARM_FEATURE_CRC32
// CHECK-V7-ALLOW-FP-INSTR-NOT: __ARM_FEATURE_NUMERIC_MAXMIN
// CHECK-V7-ALLOW-FP-INSTR-NOT: __ARM_FEATURE_DIRECTED_ROUNDING
// CHECK-V7-ALLOW-FP-INSTR: #define __ARM_FP 0xc

// RUN: %clang -target armv7ve-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V7VE %s
// CHECK-V7VE: #define __ARMEL__ 1
// CHECK-V7VE: #define __ARM_ARCH 7
// CHECK-V7VE: #define __ARM_ARCH_7VE__ 1
// CHECK-V7VE: #define __ARM_ARCH_EXT_IDIV__ 1
// CHECK-V7VE-NOT: #define __ARM_FP 0x

// RUN: %clang -target armv7ve-none-linux-gnueabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V7VE-DEFAULT-ABI-SOFT %s
// RUN: %clang -target armv7ve-none-linux-gnueabihf -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V7VE-DEFAULT-ABI-SOFT %s
// CHECK-V7VE-DEFAULT-ABI-SOFT: #define __ARMEL__ 1
// CHECK-V7VE-DEFAULT-ABI-SOFT: #define __ARM_ARCH 7
// CHECK-V7VE-DEFAULT-ABI-SOFT: #define __ARM_ARCH_7VE__ 1
// CHECK-V7VE-DEFAULT-ABI-SOFT: #define __ARM_ARCH_EXT_IDIV__ 1
// CHECK-V7VE-DEFAULT-ABI-SOFT: #define __ARM_FP 0xc

// RUN: %clang -target x86_64-apple-macosx10.10 -arch armv7s -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V7S %s
// CHECK-V7S: #define __ARMEL__ 1
// CHECK-V7S: #define __ARM_ARCH 7
// CHECK-V7S: #define __ARM_ARCH_7S__ 1
// CHECK-V7S-NOT: __ARM_FEATURE_CRC32
// CHECK-V7S-NOT: __ARM_FEATURE_NUMERIC_MAXMIN
// CHECK-V7S-NOT: __ARM_FEATURE_DIRECTED_ROUNDING
// CHECK-V7S: #define __ARM_FP 0xe

// RUN: %clang -target arm-arm-none-eabi -march=armv7-m -mfloat-abi=soft -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-VFP-FP %s
// RUN: %clang -target arm-arm-none-eabi -march=armv7-m -mfloat-abi=softfp -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-VFP-FP %s
// RUN: %clang -target arm-arm-none-eabi -march=armv7-m -mfloat-abi=hard -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-VFP-FP %s
// CHECK-VFP-FP: #define __VFP_FP__ 1

// RUN: %clang -target armv8a -mfloat-abi=hard -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-V8-BAREHF %s
// CHECK-V8-BAREHF: #define __ARMEL__ 1
// CHECK-V8-BAREHF: #define __ARM_ARCH 8
// CHECK-V8-BAREHF: #define __ARM_ARCH_8A__ 1
// CHECK-V8-BAREHF: #define __ARM_FEATURE_CRC32 1
// CHECK-V8-BAREHF: #define __ARM_FEATURE_DIRECTED_ROUNDING 1
// CHECK-V8-BAREHF: #define __ARM_FEATURE_NUMERIC_MAXMIN 1
// CHECK-V8-BAREHP: #define __ARM_FP 0xe
// CHECK-V8-BAREHF: #define __ARM_NEON__ 1
// CHECK-V8-BAREHF: #define __ARM_PCS_VFP 1
// CHECK-V8-BAREHF: #define __VFP_FP__ 1

// RUN: %clang -target armv8a -mfloat-abi=hard -mfpu=fp-armv8 -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-V8-BAREHF-FP %s
// CHECK-V8-BAREHF-FP-NOT: __ARM_NEON__ 1
// CHECK-V8-BAREHP-FP: #define __ARM_FP 0xe
// CHECK-V8-BAREHF-FP: #define __VFP_FP__ 1

// RUN: %clang -target armv8a -mfloat-abi=hard -mfpu=neon-fp-armv8 -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-V8-BAREHF-NEON-FP %s
// RUN: %clang -target armv8a -mfloat-abi=hard -mfpu=crypto-neon-fp-armv8 -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-V8-BAREHF-NEON-FP %s
// CHECK-V8-BAREHP-NEON-FP: #define __ARM_FP 0xe
// CHECK-V8-BAREHF-NEON-FP: #define __ARM_NEON__ 1
// CHECK-V8-BAREHF-NEON-FP: #define __VFP_FP__ 1

// RUN: %clang -target armv8a -mnocrc -x c -E -dM %s | FileCheck -match-full-lines --check-prefix=CHECK-V8-NOCRC %s
// CHECK-V8-NOCRC-NOT: __ARM_FEATURE_CRC32 1

// Check that -mhwdiv works properly for armv8/thumbv8 (enabled by default).

// RUN: %clang -target armv8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8 %s
// RUN: %clang -target armv8 -mthumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8 %s
// RUN: %clang -target armv8-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8 %s
// RUN: %clang -target armv8-eabi -mthumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8 %s
// V8:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target armv8 -mhwdiv=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV-V8 %s
// RUN: %clang -target armv8 -mthumb -mhwdiv=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV-V8 %s
// RUN: %clang -target armv8 -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV-V8 %s
// RUN: %clang -target armv8 -mthumb -mhwdiv=arm -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV-V8 %s
// NOHWDIV-V8-NOT:#define __ARM_ARCH_EXT_IDIV__

// RUN: %clang -target armv8a -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8A %s
// RUN: %clang -target armv8a -mthumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8A %s
// V8A:#define __ARM_ARCH_EXT_IDIV__ 1
// V8A-NOT:#define __ARM_FP 0x

// RUN: %clang -target armv8a-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8A-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8a-eabi -mthumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8A-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8a-eabihf -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8A-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8a-eabihf -mthumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8A-ALLOW-FP-INSTR %s
// V8A-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// V8A-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// RUN: %clang -target armv8m.base-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_BASELINE %s
// V8M_BASELINE: #define __ARM_ARCH 8
// V8M_BASELINE: #define __ARM_ARCH_8M_BASE__ 1
// V8M_BASELINE: #define __ARM_ARCH_EXT_IDIV__ 1
// V8M_BASELINE-NOT: __ARM_ARCH_ISA_ARM
// V8M_BASELINE: #define __ARM_ARCH_ISA_THUMB 1
// V8M_BASELINE: #define __ARM_ARCH_PROFILE 'M'
// V8M_BASELINE-NOT: __ARM_FEATURE_CRC32
// V8M_BASELINE: #define __ARM_FEATURE_CMSE 1
// V8M_BASELINE-NOT: __ARM_FEATURE_DSP
// V8M_BASELINE-NOT: __ARM_FP 0x{{.*}}
// V8M_BASELINE-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1

// RUN: %clang -target armv8m.main-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_MAINLINE %s
// V8M_MAINLINE: #define __ARM_ARCH 8
// V8M_MAINLINE: #define __ARM_ARCH_8M_MAIN__ 1
// V8M_MAINLINE: #define __ARM_ARCH_EXT_IDIV__ 1
// V8M_MAINLINE-NOT: __ARM_ARCH_ISA_ARM
// V8M_MAINLINE: #define __ARM_ARCH_ISA_THUMB 2
// V8M_MAINLINE: #define __ARM_ARCH_PROFILE 'M'
// V8M_MAINLINE-NOT: __ARM_FEATURE_CRC32
// V8M_MAINLINE: #define __ARM_FEATURE_CMSE 1
// V8M_MAINLINE-NOT: __ARM_FEATURE_DSP
// V8M_MAINLINE-NOT: #define __ARM_FP 0x
// V8M_MAINLINE: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1

// RUN: %clang -target armv8m.main-none-linux-gnueabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M-MAINLINE-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8m.main-none-linux-gnueabihf -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M-MAINLINE-ALLOW-FP-INSTR %s
// V8M-MAINLINE-ALLOW-FP-INSTR: #define __ARM_ARCH 8
// V8M-MAINLINE-ALLOW-FP-INSTR: #define __ARM_ARCH_8M_MAIN__ 1
// V8M-MAINLINE-ALLOW-FP-INSTR: #define __ARM_ARCH_EXT_IDIV__ 1
// V8M-MAINLINE-ALLOW-FP-INSTR-NOT: __ARM_ARCH_ISA_ARM
// V8M-MAINLINE-ALLOW-FP-INSTR: #define __ARM_ARCH_ISA_THUMB 2
// V8M-MAINLINE-ALLOW-FP-INSTR: #define __ARM_ARCH_PROFILE 'M'
// V8M-MAINLINE-ALLOW-FP-INSTR-NOT: __ARM_FEATURE_CRC32
// V8M-MAINLINE-ALLOW-FP-INSTR-NOT: __ARM_FEATURE_DSP
// V8M-MAINLINE-ALLOW-FP-INSTR: #define __ARM_FP 0xe
// V8M-MAINLINE-ALLOW-FP-INSTR: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1

// RUN: %clang -target arm-none-linux-gnu -march=armv8-m.main+dsp -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_MAINLINE_DSP %s
// V8M_MAINLINE_DSP: #define __ARM_ARCH 8
// V8M_MAINLINE_DSP: #define __ARM_ARCH_8M_MAIN__ 1
// V8M_MAINLINE_DSP: #define __ARM_ARCH_EXT_IDIV__ 1
// V8M_MAINLINE_DSP-NOT: __ARM_ARCH_ISA_ARM
// V8M_MAINLINE_DSP: #define __ARM_ARCH_ISA_THUMB 2
// V8M_MAINLINE_DSP: #define __ARM_ARCH_PROFILE 'M'
// V8M_MAINLINE_DSP-NOT: __ARM_FEATURE_CRC32
// V8M_MAINLINE_DSP: #define __ARM_FEATURE_DSP 1
// V8M_MAINLINE_DSP-NOT: #define __ARM_FP 0x
// V8M_MAINLINE_DSP: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1

// RUN: %clang -target arm-none-linux-gnueabi -march=armv8-m.main+dsp -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M-MAINLINE-DSP-ALLOW-FP-INSTR %s
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __ARM_ARCH 8
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __ARM_ARCH_8M_MAIN__ 1
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __ARM_ARCH_EXT_IDIV__ 1
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR-NOT: __ARM_ARCH_ISA_ARM
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __ARM_ARCH_ISA_THUMB 2
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __ARM_ARCH_PROFILE 'M'
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR-NOT: __ARM_FEATURE_CRC32
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __ARM_FEATURE_DSP 1
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __ARM_FP 0xe
// V8M-MAINLINE-DSP-ALLOW-FP-INSTR: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1

// RUN: %clang -target arm-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-DEFS %s
// CHECK-DEFS:#define __ARM_PCS 1
// CHECK-DEFS:#define __ARM_SIZEOF_MINIMAL_ENUM 4
// CHECK-DEFS:#define __ARM_SIZEOF_WCHAR_T 4

// RUN: %clang -target arm-none-linux-gnu -fno-math-errno -fno-signed-zeros\
// RUN:        -fno-trapping-math -fassociative-math -freciprocal-math\
// RUN:        -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FASTMATH %s
// RUN: %clang -target arm-none-linux-gnu -ffast-math -x c -E -dM %s -o -\
// RUN:        | FileCheck -match-full-lines --check-prefix=CHECK-FASTMATH %s
// CHECK-FASTMATH: #define __ARM_FP_FAST 1

// RUN: %clang -target arm-none-linux-gnu -fshort-wchar -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-SHORTWCHAR %s
// CHECK-SHORTWCHAR:#define __ARM_SIZEOF_WCHAR_T 2

// RUN: %clang -target arm-none-linux-gnu -fshort-enums -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-SHORTENUMS %s
// CHECK-SHORTENUMS:#define __ARM_SIZEOF_MINIMAL_ENUM 1

// Test that -mhwdiv has the right effect for a target CPU which has hwdiv enabled by default.
// RUN: %clang -target armv7 -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=HWDIV %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=HWDIV %s
// RUN: %clang -target armv7 -mcpu=cortex-a15 -mhwdiv=arm -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=HWDIV %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a15 -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=HWDIV %s
// HWDIV:#define __ARM_ARCH_EXT_IDIV__ 1

// RUN: %clang -target arm -mcpu=cortex-a15 -mhwdiv=thumb -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV %s
// RUN: %clang -target arm -mthumb -mcpu=cortex-a15 -mhwdiv=arm -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV %s
// RUN: %clang -target arm -mcpu=cortex-a15 -mhwdiv=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV %s
// RUN: %clang -target arm -mthumb -mcpu=cortex-a15 -mhwdiv=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOHWDIV %s
// NOHWDIV-NOT:#define __ARM_ARCH_EXT_IDIV__


// Check that -mfpu works properly for Cortex-A7 (enabled by default).
// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A7 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A7 %s
// RUN: %clang -target armv7-none-linux-gnueabihf -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A7 %s
// RUN: %clang -target armv7-none-linux-gnueabihf -mthumb -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A7 %s
// DEFAULTFPU-A7:#define __ARM_FP 0xe
// DEFAULTFPU-A7:#define __ARM_NEON__ 1
// DEFAULTFPU-A7:#define __ARM_VFPV4__ 1

// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a7 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A7 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a7 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A7 %s
// FPUNONE-A7-NOT:#define __ARM_FP 0x{{.*}}
// FPUNONE-A7-NOT:#define __ARM_NEON__ 1
// FPUNONE-A7-NOT:#define __ARM_VFPV4__ 1

// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a7 -mfpu=vfp4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NONEON-A7 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a7 -mfpu=vfp4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NONEON-A7 %s
// NONEON-A7:#define __ARM_FP 0xe
// NONEON-A7-NOT:#define __ARM_NEON__ 1
// NONEON-A7:#define __ARM_VFPV4__ 1

// Check that -mfpu works properly for Cortex-A5 (enabled by default).
// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A5 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A5 %s
// DEFAULTFPU-A5:#define __ARM_FP 0xe
// DEFAULTFPU-A5:#define __ARM_NEON__ 1
// DEFAULTFPU-A5:#define __ARM_VFPV4__ 1

// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a5 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A5 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a5 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A5 %s
// FPUNONE-A5-NOT:#define __ARM_FP 0x{{.*}}
// FPUNONE-A5-NOT:#define __ARM_NEON__ 1
// FPUNONE-A5-NOT:#define __ARM_VFPV4__ 1

// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a5 -mfpu=vfp4-d16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NONEON-A5 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a5 -mfpu=vfp4-d16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NONEON-A5 %s
// NONEON-A5:#define __ARM_FP 0xe
// NONEON-A5-NOT:#define __ARM_NEON__ 1
// NONEON-A5:#define __ARM_VFPV4__ 1

// FIXME: add check for further predefines
// Test whether predefines are as expected when targeting ep9312.
// RUN: %clang -target armv4t -mcpu=ep9312 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A4T %s
// A4T-NOT:#define __ARM_FEATURE_DSP
// A4T-NOT:#define __ARM_FP 0x{{.*}}

// Test whether predefines are as expected when targeting arm10tdmi.
// RUN: %clang -target armv5 -mcpu=arm10tdmi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A5T %s
// A5T-NOT:#define __ARM_FEATURE_DSP
// A5T-NOT:#define __ARM_FP 0x{{.*}}

// Test whether predefines are as expected when targeting cortex-a5i (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A5 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A5 %s
// A5:#define __ARM_ARCH 7
// A5:#define __ARM_ARCH_7A__ 1
// A5-NOT:#define __ARM_ARCH_EXT_IDIV__
// A5:#define __ARM_ARCH_PROFILE 'A'
// A5-NOT:#define __ARM_DWARF_EH__ 1
// A5-NOT: #define __ARM_FEATURE_DIRECTED_ROUNDING
// A5:#define __ARM_FEATURE_DSP 1
// A5-NOT: #define __ARM_FEATURE_NUMERIC_MAXMIN
// A5-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-a5 (softfp FP ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A5-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A5-ALLOW-FP-INSTR %s
// A5-ALLOW-FP-INSTR:#define __ARM_ARCH 7
// A5-ALLOW-FP-INSTR:#define __ARM_ARCH_7A__ 1
// A5-ALLOW-FP-INSTR-NOT:#define __ARM_ARCH_EXT_IDIV__
// A5-ALLOW-FP-INSTR:#define __ARM_ARCH_PROFILE 'A'
// A5-ALLOW-FP-INSTR-NOT: #define __ARM_FEATURE_DIRECTED_ROUNDING
// A5-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// A5-ALLOW-FP-INSTR-NOT: #define __ARM_FEATURE_NUMERIC_MAXMIN
// A5-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Test whether predefines are as expected when targeting cortex-a7 (soft FP ABI as default).
// RUN: %clang -target armv7k -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A7 %s
// RUN: %clang -target armv7k -mthumb -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A7 %s
// A7:#define __ARM_ARCH 7
// A7:#define __ARM_ARCH_EXT_IDIV__ 1
// A7:#define __ARM_ARCH_PROFILE 'A'
// A7-NOT:#define __ARM_DWARF_EH__ 1
// A7:#define __ARM_FEATURE_DSP 1
// A7-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-a7 (softfp FP ABI as default).
// RUN: %clang -target armv7k-eabi -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A7-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7k-eabi -mthumb -mcpu=cortex-a7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A7-ALLOW-FP-INSTR %s
// A7-ALLOW-FP-INSTR:#define __ARM_ARCH 7
// A7-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// A7-ALLOW-FP-INSTR:#define __ARM_ARCH_PROFILE 'A'
// A7-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// A7-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Test whether predefines are as expected when targeting cortex-a7.
// RUN: %clang -target x86_64-apple-darwin -arch armv7k -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV7K %s
// ARMV7K:#define __ARM_ARCH 7
// ARMV7K:#define __ARM_ARCH_EXT_IDIV__ 1
// ARMV7K:#define __ARM_ARCH_PROFILE 'A'
// ARMV7K:#define __ARM_DWARF_EH__ 1
// ARMV7K:#define __ARM_FEATURE_DSP 1
// ARMV7K:#define __ARM_FP 0xe
// ARMV7K:#define __ARM_PCS_VFP 1


// Test whether predefines are as expected when targeting cortex-a8 (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-a8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A8 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A8 %s
// A8-NOT:#define __ARM_ARCH_EXT_IDIV__
// A8:#define __ARM_FEATURE_DSP 1
// A8-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-a8 (softfp FP ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-a8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-a8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A8-ALLOW-FP-INSTR %s
// A8-ALLOW-FP-INSTR-NOT:#define __ARM_ARCH_EXT_IDIV__
// A8-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// A8-ALLOW-FP-INSTR:#define __ARM_FP 0xc

// Test whether predefines are as expected when targeting cortex-a9 (soft FP as default).
// RUN: %clang -target armv7 -mcpu=cortex-a9 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A9 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a9 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A9 %s
// A9-NOT:#define __ARM_ARCH_EXT_IDIV__
// A9:#define __ARM_FEATURE_DSP 1
// A9-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-a9 (softfp FP as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-a9 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A9-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-a9 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A9-ALLOW-FP-INSTR %s
// A9-ALLOW-FP-INSTR-NOT:#define __ARM_ARCH_EXT_IDIV__
// A9-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// A9-ALLOW-FP-INSTR:#define __ARM_FP 0xe


// Check that -mfpu works properly for Cortex-A12 (enabled by default).
// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a12 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A12 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a12 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A12 %s
// DEFAULTFPU-A12:#define __ARM_FP 0xe
// DEFAULTFPU-A12:#define __ARM_NEON__ 1
// DEFAULTFPU-A12:#define __ARM_VFPV4__ 1

// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a12 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A12 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a12 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A12 %s
// FPUNONE-A12-NOT:#define __ARM_FP 0x{{.*}}
// FPUNONE-A12-NOT:#define __ARM_NEON__ 1
// FPUNONE-A12-NOT:#define __ARM_VFPV4__ 1

// Test whether predefines are as expected when targeting cortex-a12 (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-a12 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A12 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a12 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A12 %s
// A12:#define __ARM_ARCH 7
// A12:#define __ARM_ARCH_7A__ 1
// A12:#define __ARM_ARCH_EXT_IDIV__ 1
// A12:#define __ARM_ARCH_PROFILE 'A'
// A12:#define __ARM_FEATURE_DSP 1
// A12-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-a12 (soft FP ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-a12 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A12-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-a12 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A12-ALLOW-FP-INSTR %s
// A12-ALLOW-FP-INSTR:#define __ARM_ARCH 7
// A12-ALLOW-FP-INSTR:#define __ARM_ARCH_7A__ 1
// A12-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// A12-ALLOW-FP-INSTR:#define __ARM_ARCH_PROFILE 'A'
// A12-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// A12-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Test whether predefines are as expected when targeting cortex-a15 (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A15 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A15 %s
// A15:#define __ARM_ARCH_EXT_IDIV__ 1
// A15:#define __ARM_FEATURE_DSP 1
// A15-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-a15 (softfp ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A15-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-a15 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A15-ALLOW-FP-INSTR %s
// A15-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// A15-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// A15-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Check that -mfpu works properly for Cortex-A17 (enabled by default).
// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a17 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A17 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a17 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=DEFAULTFPU-A17 %s
// DEFAULTFPU-A17:#define __ARM_FP 0xe
// DEFAULTFPU-A17:#define __ARM_NEON__ 1
// DEFAULTFPU-A17:#define __ARM_VFPV4__ 1

// RUN: %clang -target armv7-none-linux-gnueabi -mcpu=cortex-a17 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A17 %s
// RUN: %clang -target armv7-none-linux-gnueabi -mthumb -mcpu=cortex-a17 -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=FPUNONE-A17 %s
// FPUNONE-A17-NOT:#define __ARM_FP 0x{{.*}}
// FPUNONE-A17-NOT:#define __ARM_NEON__ 1
// FPUNONE-A17-NOT:#define __ARM_VFPV4__ 1

// Test whether predefines are as expected when targeting cortex-a17 (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-a17 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A17 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-a17 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A17 %s
// A17:#define __ARM_ARCH 7
// A17:#define __ARM_ARCH_7A__ 1
// A17:#define __ARM_ARCH_EXT_IDIV__ 1
// A17:#define __ARM_ARCH_PROFILE 'A'
// A17:#define __ARM_FEATURE_DSP 1
// A17-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-a17 (softfp FP ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-a17 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A17-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-a17 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=A17-ALLOW-FP-INSTR %s
// A17-ALLOW-FP-INSTR:#define __ARM_ARCH 7
// A17-ALLOW-FP-INSTR:#define __ARM_ARCH_7A__ 1
// A17-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// A17-ALLOW-FP-INSTR:#define __ARM_ARCH_PROFILE 'A'
// A17-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// A17-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Test whether predefines are as expected when targeting swift (soft FP ABI as default).
// RUN: %clang -target armv7s -mcpu=swift -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=SWIFT %s
// RUN: %clang -target armv7s -mthumb -mcpu=swift -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=SWIFT %s
// SWIFT:#define __ARM_ARCH_EXT_IDIV__ 1
// SWIFT:#define __ARM_FEATURE_DSP 1
// SWIFT-NOT:#define __ARM_FP 0xxE

// Test whether predefines are as expected when targeting swift (softfp FP ABI as default).
// RUN: %clang -target armv7s-eabi -mcpu=swift -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=SWIFT-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7s-eabi -mthumb -mcpu=swift -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=SWIFT-ALLOW-FP-INSTR %s
// SWIFT-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// SWIFT-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// SWIFT-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Test whether predefines are as expected when targeting ARMv8-A Cortex implementations (soft FP ABI as default)
// RUN: %clang -target armv8 -mcpu=cortex-a32 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=cortex-a32 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mcpu=cortex-a35 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=cortex-a35 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mcpu=cortex-a53 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=cortex-a53 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mcpu=cortex-a57 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=cortex-a57 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mcpu=cortex-a72 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=cortex-a72 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mcpu=cortex-a73 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=cortex-a73 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
//
// RUN: %clang -target armv8 -mcpu=exynos-m3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=exynos-m3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mcpu=exynos-m4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=exynos-m4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mcpu=exynos-m5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// RUN: %clang -target armv8 -mthumb -mcpu=exynos-m5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8 %s
// ARMV8:#define __ARM_ARCH_EXT_IDIV__ 1
// ARMV8:#define __ARM_FEATURE_DSP 1
// ARMV8-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting ARMv8-A Cortex implementations (softfp FP ABI as default)
// RUN: %clang -target armv8-eabi -mcpu=cortex-a32 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=cortex-a32 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mcpu=cortex-a35 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=cortex-a35 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mcpu=cortex-a53 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=cortex-a53 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mcpu=cortex-a57 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=cortex-a57 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mcpu=cortex-a72 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=cortex-a72 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mcpu=cortex-a73 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=cortex-a73 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
//
// RUN: %clang -target armv8-eabi -mcpu=exynos-m3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=exynos-m3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mcpu=exynos-m4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=exynos-m4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mcpu=exynos-m5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv8-eabi -mthumb -mcpu=exynos-m5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=ARMV8-ALLOW-FP-INSTR %s
// ARMV8-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// ARMV8-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// ARMV8-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Test whether predefines are as expected when targeting cortex-r4.
// RUN: %clang -target armv7 -mcpu=cortex-r4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R4-ARM %s
// R4-ARM-NOT:#define __ARM_ARCH_EXT_IDIV__
// R4-ARM:#define __ARM_FEATURE_DSP 1
// R4-ARM-NOT:#define __ARM_FP 0x{{.*}}

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-r4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R4-THUMB %s
// R4-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1
// R4-THUMB:#define __ARM_FEATURE_DSP 1
// R4-THUMB-NOT:#define __ARM_FP 0x{{.*}}

// Test whether predefines are as expected when targeting cortex-r4f (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-r4f -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R4F-ARM %s
// R4F-ARM-NOT:#define __ARM_ARCH_EXT_IDIV__
// R4F-ARM:#define __ARM_FEATURE_DSP 1
// R4F-ARM-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-r4f (softfp FP ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-r4f -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R4F-ARM-ALLOW-FP-INSTR %s
// R4F-ARM-ALLOW-FP-INSTR-NOT:#define __ARM_ARCH_EXT_IDIV__
// R4F-ARM-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// R4F-ARM-ALLOW-FP-INSTR:#define __ARM_FP 0xc

// RUN: %clang -target armv7 -mthumb -mcpu=cortex-r4f -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R4F-THUMB %s
// R4F-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1
// R4F-THUMB:#define __ARM_FEATURE_DSP 1
// R4F-THUMB-NOT:#define __ARM_FP 0x

// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-r4f -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R4F-THUMB-ALLOW-FP-INSTR %s
// R4F-THUMB-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// R4F-THUMB-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// R4F-THUMB-ALLOW-FP-INSTR:#define __ARM_FP 0xc

// Test whether predefines are as expected when targeting cortex-r5 (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-r5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R5 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-r5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R5 %s
// R5:#define __ARM_ARCH_EXT_IDIV__ 1
// R5:#define __ARM_FEATURE_DSP 1
// R5-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-r5 (softfp FP ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-r5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R5-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-r5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R5-ALLOW-FP-INSTR %s
// R5-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// R5-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// R5-ALLOW-FP-INSTR:#define __ARM_FP 0xc

// Test whether predefines are as expected when targeting cortex-r7 and cortex-r8 (soft FP ABI as default).
// RUN: %clang -target armv7 -mcpu=cortex-r7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-r7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8 %s
// RUN: %clang -target armv7 -mcpu=cortex-r8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8 %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-r8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8 %s
// R7-R8:#define __ARM_ARCH_EXT_IDIV__ 1
// R7-R8:#define __ARM_FEATURE_DSP 1
// R7-R8-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-r7 and cortex-r8 (softfp FP ABI as default).
// RUN: %clang -target armv7-eabi -mcpu=cortex-r7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-r7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mcpu=cortex-r8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-r8 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=R7-R8-ALLOW-FP-INSTR %s
// R7-R8-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// R7-R8-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// R7-R8-ALLOW-FP-INSTR:#define __ARM_FP 0xe

// Test whether predefines are as expected when targeting cortex-m0.
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m0 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M0-THUMB %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m0plus -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M0-THUMB %s
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m1 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M0-THUMB %s
// RUN: %clang -target armv7 -mthumb -mcpu=sc000 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M0-THUMB %s
// M0-THUMB-NOT:#define __ARM_ARCH_EXT_IDIV__
// M0-THUMB-NOT:#define __ARM_FEATURE_DSP
// M0-THUMB-NOT:#define __ARM_FP 0x{{.*}}

// Test whether predefines are as expected when targeting cortex-m3.
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m3 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M3-THUMB %s
// RUN: %clang -target armv7 -mthumb -mcpu=sc300 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M3-THUMB %s
// M3-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1
// M3-THUMB-NOT:#define __ARM_FEATURE_DSP
// M3-THUMB-NOT:#define __ARM_FP 0x{{.*}}

// Test whether predefines are as expected when targeting cortex-m4 (soft FP ABI as default).
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M4-THUMB %s
// M4-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1
// M4-THUMB:#define __ARM_FEATURE_DSP 1
// M4-THUMB-NOT:#define __ARM_FP 0x

// Test whether predefines are as expected when targeting cortex-m4 (softfp ABI as default).
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-m4 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M4-THUMB-ALLOW-FP-INSTR %s
// M4-THUMB-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// M4-THUMB-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// M4-THUMB-ALLOW-FP-INSTR:#define __ARM_FP 0x6

// Test whether predefines are as expected when targeting cortex-m7 (soft FP ABI as default).
// RUN: %clang -target armv7 -mthumb -mcpu=cortex-m7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M7-THUMB %s
// M7-THUMB:#define __ARM_ARCH_EXT_IDIV__ 1
// M7-THUMB:#define __ARM_FEATURE_DSP 1
// M7-THUMB-NOT:#define __ARM_FP 0x
// M7-THUMB-NOT:#define __ARM_FPV5__

// Test whether predefines are as expected when targeting cortex-m7 (softfp FP ABI as default).
// RUN: %clang -target armv7-eabi -mthumb -mcpu=cortex-m7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M7-THUMB-ALLOW-FP-INSTR %s
// M7-THUMB-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// M7-THUMB-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// M7-THUMB-ALLOW-FP-INSTR:#define __ARM_FP 0xe
// M7-THUMB-ALLOW-FP-INSTR:#define __ARM_FPV5__ 1

// Check that -mcmse (security extension) option works correctly for v8-M targets
// RUN: %clang -target armv8m.base-none-linux-gnu -mcmse -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_CMSE %s
// RUN: %clang -target armv8m.main-none-linux-gnu -mcmse -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_CMSE %s
// RUN: %clang -target arm-none-linux-gnu -mcpu=cortex-m33 -mcmse -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_CMSE %s
// RUN: %clang -target arm -mcpu=cortex-m23 -mcmse -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_CMSE %s
// RUN: %clang -target arm-none-linux-gnu -mcpu=cortex-m55 -mcmse -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=V8M_CMSE %s
// V8M_CMSE-NOT: __ARM_FEATURE_CMSE 1
// V8M_CMSE: #define __ARM_FEATURE_CMSE 3

// Check that CMSE is not defined on architectures w/o support for security extension
// RUN: %clang -target arm-arm-none-gnueabi -mcpu=cortex-a5 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOTV8M_CMSE %s
// RUN: %clang -target armv8a-none-linux-gnu -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOTV8M_CMSE %s
// RUN: %clang -target armv8.1a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=NOTV8M_CMSE %s
// NOTV8M_CMSE-NOT: __ARM_FEATURE_CMSE

// Check that -mcmse option gives error on non v8-M targets
// RUN: not %clang -target arm-arm-none-eabi -mthumb -mcmse -mcpu=cortex-m7 -x c -E -dM %s -o - 2>&1 | FileCheck -match-full-lines --check-prefix=NOTV8MCMSE_OPT %s
// NOTV8MCMSE_OPT: error: -mcmse is not supported for cortex-m7

// Test whether predefines are as expected when targeting v8m cores
// RUN: %clang -target arm -mcpu=cortex-m23 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M23 %s
// M23: #define __ARM_ARCH 8
// M23: #define __ARM_ARCH_8M_BASE__ 1
// M23: #define __ARM_ARCH_EXT_IDIV__ 1
// M23-NOT: __ARM_ARCH_ISA_ARM
// M23: #define __ARM_ARCH_ISA_THUMB 1
// M23: #define __ARM_ARCH_PROFILE 'M'
// M23-NOT: __ARM_FEATURE_CRC32
// M23-NOT: __ARM_FEATURE_DSP
// M23-NOT: __ARM_FP 0x{{.*}}
// M23-NOT: __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1

// Test whether predefines are as expected when targeting m33 (soft FP ABI as default).
// RUN: %clang -target arm -mcpu=cortex-m33 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M33 %s
// M33: #define __ARM_ARCH 8
// M33: #define __ARM_ARCH_8M_MAIN__ 1
// M33: #define __ARM_ARCH_EXT_IDIV__ 1
// M33-NOT: __ARM_ARCH_ISA_ARM
// M33: #define __ARM_ARCH_ISA_THUMB 2
// M33: #define __ARM_ARCH_PROFILE 'M'
// M33-NOT: __ARM_FEATURE_CRC32
// M33: #define __ARM_FEATURE_DSP 1
// M33-NOT: #define __ARM_FP 0x
// M33: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1

// Test whether predefines are as expected when targeting m33 (softfp FP ABI as default).
// RUN: %clang -target arm-eabi -mcpu=cortex-m33 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M33-ALLOW-FP-INSTR %s
// M33-ALLOW-FP-INSTR: #define __ARM_ARCH 8
// M33-ALLOW-FP-INSTR: #define __ARM_ARCH_8M_MAIN__ 1
// M33-ALLOW-FP-INSTR: #define __ARM_ARCH_EXT_IDIV__ 1
// M33-ALLOW-FP-INSTR-NOT: __ARM_ARCH_ISA_ARM
// M33-ALLOW-FP-INSTR: #define __ARM_ARCH_ISA_THUMB 2
// M33-ALLOW-FP-INSTR: #define __ARM_ARCH_PROFILE 'M'
// M33-ALLOW-FP-INSTR-NOT: __ARM_FEATURE_CRC32
// M33-ALLOW-FP-INSTR: #define __ARM_FEATURE_DSP 1
// M33-ALLOW-FP-INSTR: #define __ARM_FP 0x6
// M33-ALLOW-FP-INSTR: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1

// Test whether predefines are as expected when targeting cortex-m55 (softfp FP ABI as default).
// RUN: %clang -target arm-eabi -mcpu=cortex-m55 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=M55 %s
// M55: #define __ARM_ARCH 8
// M55: #define __ARM_ARCH_8_1M_MAIN__ 1
// M55: #define __ARM_ARCH_EXT_IDIV__ 1
// M55-NOT: __ARM_ARCH_ISA_ARM
// M55: #define __ARM_ARCH_ISA_THUMB 2
// M55: #define __ARM_ARCH_PROFILE 'M'
// M55-NOT: __ARM_FEATURE_CRC32
// M55: #define __ARM_FEATURE_DSP 1
// M55: #define __ARM_FEATURE_MVE 3
// M55: #define __ARM_FP 0xe
// M55: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1

// Test whether predefines are as expected when targeting krait (soft FP as default).
// RUN: %clang -target armv7 -mcpu=krait -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=KRAIT %s
// RUN: %clang -target armv7 -mthumb -mcpu=krait -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=KRAIT %s
// KRAIT:#define __ARM_ARCH_EXT_IDIV__ 1
// KRAIT:#define __ARM_FEATURE_DSP 1
// KRAIT-NOT:#define  __ARM_VFPV4__

// Test whether predefines are as expected when targeting krait (softfp FP as default).
// RUN: %clang -target armv7-eabi -mcpu=krait -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=KRAIT-ALLOW-FP-INSTR %s
// RUN: %clang -target armv7-eabi -mthumb -mcpu=krait -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=KRAIT-ALLOW-FP-INSTR %s
// KRAIT-ALLOW-FP-INSTR:#define __ARM_ARCH_EXT_IDIV__ 1
// KRAIT-ALLOW-FP-INSTR:#define __ARM_FEATURE_DSP 1
// KRAIT-ALLOW-FP-INSTR:#define  __ARM_VFPV4__ 1

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M %s
// CHECK-V81M: #define __ARM_ARCH 8
// CHECK-V81M: #define __ARM_ARCH_8_1M_MAIN__ 1
// CHECK-V81M: #define __ARM_ARCH_ISA_THUMB 2
// CHECK-V81M: #define __ARM_ARCH_PROFILE 'M'
// CHECK-V81M-NOT: #define __ARM_FEATURE_DSP
// CHECK-V81M-NOT: #define __ARM_FEATURE_MVE
// CHECK-V81M-NOT: #define __ARM_FEATURE_SIMD32

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVE %s
// CHECK-V81M-MVE: #define __ARM_FEATURE_DSP 1
// CHECK-V81M-MVE: #define __ARM_FEATURE_MVE 1
// CHECK-V81M-MVE: #define __ARM_FEATURE_SIMD32 1

// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVEFP %s
// CHECK-V81M-MVEFP: #define __ARM_FEATURE_DSP 1
// CHECK-V81M-MVEFP: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-V81M-MVEFP: #define __ARM_FEATURE_MVE 3
// CHECK-V81M-MVEFP: #define __ARM_FEATURE_SIMD32 1
// CHECK-V81M-MVEFP: #define __ARM_FPV5__ 1

// fpu=none/nofp discards mve.fp, but not mve/dsp
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp+nofp            -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVEFP-NOFP %s
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp      -mfpu=none -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVEFP-NOFP %s
// CHECK-V81M-MVEFP-NOFP: #define __ARM_FEATURE_DSP 1
// CHECK-V81M-MVEFP-NOFP: #define __ARM_FEATURE_MVE 1

// nomve discards mve.fp
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp+nomve -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVEFP-NOMVE %s
// CHECK-V81M-MVEFP-NOMVE-NOT: #define __ARM_FEATURE_MVE

// mve+fp doesn't imply mve.fp
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve+fp -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVE-FP %s
// CHECK-V81M-MVE-FP: #define __ARM_FEATURE_MVE 1

// nodsp discards both dsp and mve ...
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve+nodsp -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVE-NODSP %s
// CHECK-V81M-MVE-NODSP-NOT: #define __ARM_FEATURE_MVE
// CHECK-V81M-MVE-NODSP-NOT: #define __ARM_FEATURE_DSP

// ... and also mve.fp
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main+mve.fp+nodsp -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81M-MVE-NODSP %s
// CHECK-V81M-MVE-NODSP-NOT: #define __ARM_FEATURE_MVE
// CHECK-V81M-MVE-NODSP-NOT: #define __ARM_FEATURE_DSP

// Test CDE (Custom Datapath Extension) feature test macros

// RUN: %clang -target arm-arm-none-eabi -march=armv8m.main -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-V8M-NOCDE %s
// CHECK-V8M-NOCDE-NOT: #define __ARM_FEATURE_CDE
// CHECK-V8M-NOCDE-NOT: #define __ARM_FEATURE_CDE_COPROC
// RUN: %clang -target arm-arm-none-eabi -march=armv8m.main+cdecp0+cdecp1+cdecp7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8M-CDE-MASK1 %s
// CHECK-V8M-CDE-MASK1: #define __ARM_FEATURE_CDE 1
// CHECK-V8M-CDE-MASK1: #define __ARM_FEATURE_CDE_COPROC 0x83
// RUN: %clang -target arm-arm-none-eabi -march=armv8m.main+cdecp0+cdecp1+cdecp2+cdecp3+cdecp4+cdecp5+cdecp6+cdecp7 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V8M-CDE-MASK2 %s
// CHECK-V8M-CDE-MASK2: #define __ARM_FEATURE_CDE 1
// CHECK-V8M-CDE-MASK2: #define __ARM_FEATURE_CDE_COPROC 0xff

// RUN: %clang -target armv8.1a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V81A %s
// CHECK-V81A: #define __ARM_ARCH 8
// CHECK-V81A: #define __ARM_ARCH_8_1A__ 1
// CHECK-V81A: #define __ARM_ARCH_PROFILE 'A'
// CHECK-V81A: #define __ARM_FEATURE_QRDMX 1
// CHECK-V81A: #define __ARM_FP 0xe

// RUN: %clang -target armv8.2a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V82A %s
// CHECK-V82A: #define __ARM_ARCH 8
// CHECK-V82A: #define __ARM_ARCH_8_2A__ 1
// CHECK-V82A: #define __ARM_ARCH_PROFILE 'A'
// CHECK-V82A: #define __ARM_FEATURE_QRDMX 1
// CHECK-V82A: #define __ARM_FP 0xe

// RUN: %clang -target armv8.3a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V83A %s
// CHECK-V83A: #define __ARM_ARCH 8
// CHECK-V83A: #define __ARM_ARCH_8_3A__ 1
// CHECK-V83A: #define __ARM_ARCH_PROFILE 'A'

// RUN: %clang -target armv8.4a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V84A %s
// CHECK-V84A: #define __ARM_ARCH 8
// CHECK-V84A: #define __ARM_ARCH_8_4A__ 1
// CHECK-V84A: #define __ARM_ARCH_PROFILE 'A'

// RUN: %clang -target armv8.5a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V85A %s
// CHECK-V85A: #define __ARM_ARCH 8
// CHECK-V85A: #define __ARM_ARCH_8_5A__ 1
// CHECK-V85A: #define __ARM_ARCH_PROFILE 'A'

// RUN: %clang -target armv8.6a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V86A %s
// CHECK-V86A: #define __ARM_ARCH 8
// CHECK-V86A: #define __ARM_ARCH_8_6A__ 1
// CHECK-V86A: #define __ARM_ARCH_PROFILE 'A'

// RUN: %clang -target armv8.7a-none-none-eabi -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-V87A %s
// CHECK-V87A: #define __ARM_ARCH 8
// CHECK-V87A: #define __ARM_ARCH_8_7A__ 1
// CHECK-V87A: #define __ARM_ARCH_PROFILE 'A'

// RUN: %clang -target arm-none-none-eabi -march=armv7-m -mfpu=softvfp -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SOFTVFP %s
// CHECK-SOFTVFP-NOT: #define __ARM_FP 0x

// ================== Check BFloat16 Extensions.
// RUN: %clang -target arm-arm-none-eabi -march=armv8.6-a+bf16 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-BFLOAT %s
// CHECK-BFLOAT: #define __ARM_BF16_FORMAT_ALTERNATIVE 1
// CHECK-BFLOAT: #define __ARM_FEATURE_BF16 1
// CHECK-BFLOAT: #define __ARM_FEATURE_BF16_VECTOR_ARITHMETIC 1

// Check crypto feature test macros
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+crypto -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-CRYPTO %s
// CHECK-CRYPTO: #define __ARM_ARCH_PROFILE 'A'
// CHECK-CRYPTO: #define __ARM_FEATURE_AES 1
// CHECK-CRYPTO: #define __ARM_FEATURE_CRYPTO 1
// CHECK-CRYPTO: #define __ARM_FEATURE_SHA2 1
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+nocrypto -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-NOCRYPTO %s
// CHECK-NOCRYPTO: #define __ARM_ARCH_PROFILE 'A'
// CHECK-NOCRYPTO-NOT: #define __ARM_FEATURE_AES 1
// CHECK-NOCRYPTO-NOT: #define __ARM_FEATURE_CRYPTO 1
// CHECK-NOCRYPTO-NOT: #define __ARM_FEATURE_SHA2 1
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+aes+nosha2 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-AES %s
// CHECK-AES: #define __ARM_ARCH_PROFILE 'A'
// CHECK-AES: #define __ARM_FEATURE_AES 1
// CHECK-AES-NOT: #define __ARM_FEATURE_CRYPTO 1
// CHECK-AES-NOT: #define __ARM_FEATURE_SHA2 1
// RUN: %clang -target arm-arm-none-eabi -march=armv8-a+noaes+sha2 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-SHA2 %s
// CHECK-SHA2: #define __ARM_ARCH_PROFILE 'A'
// CHECK-SHA2-NOT: #define __ARM_FEATURE_AES 1
// CHECK-SHA2-NOT: #define __ARM_FEATURE_CRYPTO 1
// CHECK-SHA2: #define __ARM_FEATURE_SHA2 1
