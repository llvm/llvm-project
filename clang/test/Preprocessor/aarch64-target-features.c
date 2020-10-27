// RUN: %clang -target aarch64-none-linux-gnu -x c -E -dM %s -o - | FileCheck %s
// RUN: %clang -target arm64-none-linux-gnu -x c -E -dM %s -o - | FileCheck %s

// CHECK: __AARCH64EL__ 1
// CHECK: __ARM_64BIT_STATE 1
// CHECK-NOT: __ARM_32BIT_STATE
// CHECK: __ARM_ACLE 200
// CHECK: __ARM_ALIGN_MAX_STACK_PWR 4
// CHECK: __ARM_ARCH 8
// CHECK: __ARM_ARCH_ISA_A64 1
// CHECK-NOT: __ARM_ARCH_ISA_ARM
// CHECK-NOT: __ARM_ARCH_ISA_THUMB
// CHECK-NOT: __ARM_FEATURE_QBIT
// CHECK-NOT: __ARM_FEATURE_DSP
// CHECK-NOT: __ARM_FEATURE_SAT
// CHECK-NOT: __ARM_FEATURE_SIMD32
// CHECK: __ARM_ARCH_PROFILE 'A'
// CHECK-NOT: __ARM_FEATURE_BIG_ENDIAN
// CHECK: __ARM_FEATURE_CLZ 1
// CHECK-NOT: __ARM_FEATURE_CRC32 1
// CHECK-NOT: __ARM_FEATURE_CRYPTO 1
// CHECK: __ARM_FEATURE_DIRECTED_ROUNDING 1
// CHECK: __ARM_FEATURE_DIV 1
// CHECK: __ARM_FEATURE_FMA 1
// CHECK: __ARM_FEATURE_IDIV 1
// CHECK: __ARM_FEATURE_LDREX 0xF
// CHECK: __ARM_FEATURE_NUMERIC_MAXMIN 1
// CHECK: __ARM_FEATURE_UNALIGNED 1
// CHECK: __ARM_FP 0xE
// CHECK: __ARM_FP16_ARGS 1
// CHECK: __ARM_FP16_FORMAT_IEEE 1
// CHECK-NOT: __ARM_FP_FAST 1
// CHECK: __ARM_NEON 1
// CHECK: __ARM_NEON_FP 0xE
// CHECK: __ARM_PCS_AAPCS64 1
// CHECK-NOT: __ARM_PCS 1
// CHECK-NOT: __ARM_PCS_VFP 1
// CHECK-NOT: __ARM_SIZEOF_MINIMAL_ENUM 1
// CHECK-NOT: __ARM_SIZEOF_WCHAR_T 2
// CHECK-NOT: __ARM_FEATURE_SVE
// CHECK-NOT: __ARM_FEATURE_DOTPROD
// CHECK-NOT: __ARM_FEATURE_PAC_DEFAULT
// CHECK-NOT: __ARM_FEATURE_BTI_DEFAULT
// CHECK-NOT: __ARM_BF16_FORMAT_ALTERNATIVE 1
// CHECK-NOT: __ARM_FEATURE_BF16 1
// CHECK-NOT: __ARM_FEATURE_BF16_VECTOR_ARITHMETIC 1
// CHECK-NOT: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 0
// CHECK-NOT: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 128
// CHECK-NOT: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 256
// CHECK-NOT: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 512
// CHECK-NOT: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 1024
// CHECK-NOT: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 2048

// RUN: %clang -target aarch64_be-eabi -x c -E -dM %s -o - | FileCheck %s -check-prefix CHECK-BIGENDIAN
// CHECK-BIGENDIAN: __ARM_BIG_ENDIAN 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+crypto -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRYPTO %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a+crypto -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRYPTO %s
// CHECK-CRYPTO: __ARM_FEATURE_CRYPTO 1

// RUN: %clang -target aarch64-none-linux-gnu -mcrc -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRC32 %s
// RUN: %clang -target arm64-none-linux-gnu -mcrc -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRC32 %s
// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+crc -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRC32 %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a+crc -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-CRC32 %s
// CHECK-CRC32: __ARM_FEATURE_CRC32 1

// RUN: %clang -target aarch64-none-linux-gnu -fno-math-errno -fno-signed-zeros\
// RUN:        -fno-trapping-math -fassociative-math -freciprocal-math\
// RUN:        -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FASTMATH %s
// RUN: %clang -target aarch64-none-linux-gnu -ffast-math -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FASTMATH %s
// RUN: %clang -target arm64-none-linux-gnu -ffast-math -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-FASTMATH %s
// CHECK-FASTMATH: __ARM_FP_FAST 1

// RUN: %clang -target aarch64-none-linux-gnu -fshort-wchar -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTWCHAR %s
// RUN: %clang -target arm64-none-linux-gnu -fshort-wchar -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTWCHAR %s
// CHECK-SHORTWCHAR: __ARM_SIZEOF_WCHAR_T 2

// RUN: %clang -target aarch64-none-linux-gnu -fshort-enums -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTENUMS %s
// RUN: %clang -target arm64-none-linux-gnu -fshort-enums -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SHORTENUMS %s
// CHECK-SHORTENUMS: __ARM_SIZEOF_MINIMAL_ENUM 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+simd -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-NEON %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a+simd -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: __ARM_NEON 1
// CHECK-NEON: __ARM_NEON_FP 0xE

// RUN: %clang -target aarch64-none-eabi -march=armv8.1-a -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-QRDMX %s
// RUN: %clang -target aarch64-none-eabi -march=armv8.2-a -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-QRDMX %s
// CHECK-QRDMX: __ARM_FEATURE_QRDMX 1

// RUN: %clang -target aarch64 -march=arm64 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-ARCH-NOT-ACCEPT %s
// RUN: %clang -target aarch64 -march=aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-ARCH-NOT-ACCEPT %s
// CHECK-ARCH-NOT-ACCEPT: error: the clang compiler does not support

// RUN: %clang -target aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-GENERIC %s
// RUN: %clang -target aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-GENERIC %s
// CHECK-GENERIC: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon"

// RUN: %clang -target aarch64 -mtune=cyclone -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MTUNE-CYCLONE %s

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE %s
// CHECK-SVE: __ARM_FEATURE_SVE 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve+bf16 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE-BF16 %s
// CHECK-SVE-BF16: __ARM_FEATURE_BF16_SCALAR_ARITHMETIC 1
// CHECK-SVE-BF16: __ARM_FEATURE_SVE 1
// CHECK-SVE-BF16: __ARM_FEATURE_SVE_BF16 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve+i8mm -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE-I8MM %s
// CHECK-SVE-I8MM: __ARM_FEATURE_SVE 1
// CHECK-SVE-I8MM: __ARM_FEATURE_SVE_MATMUL_INT8 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve+f32mm -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE-F32MM %s
// CHECK-SVE-F32MM: __ARM_FEATURE_SVE 1
// CHECK-SVE-F32MM: __ARM_FEATURE_SVE_MATMUL_FP32 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve+f64mm -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE-F64MM %s
// CHECK-SVE-F64MM: __ARM_FEATURE_SVE 1
// CHECK-SVE-F64MM: __ARM_FEATURE_SVE_MATMUL_FP64 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8.5-a+sve -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE-8_5 %s
// CHECK-SVE-8_5-NOT: __ARM_FEATURE_SVE_BF16 1
// CHECK-SVE-8_5-NOT: __ARM_FEATURE_SVE_MATMUL_FP32 1
// CHECK-SVE-8_5-NOT: __ARM_FEATURE_SVE_MATMUL_INT8 1
// CHECK-SVE-8_5: __ARM_FEATURE_SVE 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8.6-a+sve -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE-8_6 %s
// CHECK-SVE-8_6: __ARM_FEATURE_SVE 1
// CHECK-SVE-8_6: __ARM_FEATURE_SVE_BF16 1
// CHECK-SVE-8_6: __ARM_FEATURE_SVE_MATMUL_FP32 1
// CHECK-SVE-8_6: __ARM_FEATURE_SVE_MATMUL_INT8 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8.6-a+sve+noi8mm+nobf16+nof32mm -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE-8_6-NOFEATURES %s
// CHECK-SVE-8_6-NOFEATURES-NOT: __ARM_FEATURE_SVE_BF16 1
// CHECK-SVE-8_6-NOFEATURES-NOT: __ARM_FEATURE_SVE_MATMUL_FP32 1
// CHECK-SVE-8_6-NOFEATURES-NOT: __ARM_FEATURE_SVE_MATMUL_INT8 1
// CHECK-SVE-8_6-NOFEATURES:     __ARM_FEATURE_SVE 1

// The following tests may need to be revised in the future since
// SVE2 is currently still part of Future Architecture Technologies
// (https://developer.arm.com/docs/ddi0602/latest)
//
// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve2 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE2 %s
// CHECK-SVE2: __ARM_FEATURE_SVE2 1
// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve2-aes -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE2AES %s
// CHECK-SVE2AES: __ARM_FEATURE_SVE2_AES 1
// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve2-sha3 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE2SHA3 %s
// CHECK-SVE2SHA3: __ARM_FEATURE_SVE2_SHA3 1
// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve2-sm4 -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE2SM4 %s
// CHECK-SVE2SM4: __ARM_FEATURE_SVE2_SM4 1
// RUN: %clang -target aarch64-none-linux-gnu -march=armv8-a+sve2-bitperm -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-SVE2BITPERM %s
// CHECK-SVE2BITPERM: __ARM_FEATURE_SVE2_BITPERM 1

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8.2a+dotprod -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-DOTPROD %s
// CHECK-DOTPROD: __ARM_FEATURE_DOTPROD 1

// On ARMv8.2-A and above, +fp16fml implies +fp16.
// On ARMv8.4-A and above, +fp16 implies +fp16fml.
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2-a+nofp16fml+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2-a+nofp16+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-FML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2-a+fp16+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8-a+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-FML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8-a+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+nofp16fml+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-FML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+nofp16+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-FML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+fp16+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+fp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-FML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-FML --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// CHECK-FULLFP16-FML:           #define __ARM_FEATURE_FP16FML 1
// CHECK-FULLFP16-NOFML-NOT:     #define __ARM_FEATURE_FP16FML 1
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FP 0xE
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FP16_FORMAT_IEEE 1

// +fp16fml+nosimd doesn't make sense as the fp16fml instructions all require SIMD.
// However, as +fp16fml implies +fp16 there is a set of defines that we would expect.
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8-a+fp16fml+nosimd -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8-a+fp16+nosimd -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+fp16fml+nosimd -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+fp16+nosimd -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// CHECK-FULLFP16-SCALAR-NOT:   #define __ARM_FEATURE_FP16FML 1
// CHECK-FULLFP16-SCALAR:       #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-SCALAR-NOT:   #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-SCALAR:       #define __ARM_FP 0xE
// CHECK-FULLFP16-SCALAR:       #define __ARM_FP16_FORMAT_IEEE 1

// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2-a -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2-a+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2-a+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2-a+fp16fml+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+nofp16fml -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.4-a+fp16fml+nofp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-NOFML-VECTOR-SCALAR %s
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR-NOT: #define __ARM_FEATURE_FP16FML 1
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR-NOT: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR-NOT: #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR:     #define __ARM_FP 0xE
// CHECK-FULLFP16-NOFML-VECTOR-SCALAR:     #define __ARM_FP16_FORMAT_IEEE 1

// ================== Check whether -mtune accepts mixed-case features.
// RUN: %clang -target aarch64 -mtune=CYCLONE -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MTUNE-CYCLONE %s
// CHECK-MTUNE-CYCLONE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+zcm" "-target-feature" "+zcz"

// RUN: %clang -target aarch64 -mcpu=apple-a7 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A7 %s
// RUN: %clang -target aarch64 -mcpu=apple-a8 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A7 %s
// RUN: %clang -target aarch64 -mcpu=apple-a9 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A7 %s
// RUN: %clang -target aarch64 -mcpu=apple-a10 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A10 %s
// RUN: %clang -target aarch64 -mcpu=apple-a11 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A11 %s
// RUN: %clang -target aarch64 -mcpu=apple-a12 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A12 %s
// RUN: %clang -target aarch64 -mcpu=apple-a13 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A13 %s
// RUN: %clang -target aarch64 -mcpu=apple-s4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A12 %s
// RUN: %clang -target aarch64 -mcpu=apple-s5 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A12 %s
// RUN: %clang -target aarch64 -mcpu=cyclone -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-APPLE-A7 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A34 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A35 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A53 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A57 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A72 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-CORTEX-A73 %s
// RUN: %clang -target aarch64 -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-M1 %s
// RUN: %clang -target aarch64 -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-M4 %s
// RUN: %clang -target aarch64 -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-M4 %s
// RUN: %clang -target aarch64 -mcpu=kryo -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-KRYO %s
// RUN: %clang -target aarch64 -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-THUNDERX2T99 %s
// RUN: %clang -target aarch64 -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A64FX %s
// RUN: %clang -target aarch64 -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-CARMEL %s
// CHECK-MCPU-APPLE-A7: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crypto" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+sha2" "-target-feature" "+aes"
// CHECK-MCPU-APPLE-A10: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+rdm" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+sha2" "-target-feature" "+aes"
// CHECK-MCPU-APPLE-A11: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+ras" "-target-feature" "+lse" "-target-feature" "+rdm" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+sha2" "-target-feature" "+aes"
// CHECK-MCPU-APPLE-A12: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.3a" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+fullfp16" "-target-feature" "+ras" "-target-feature" "+lse" "-target-feature" "+rdm" "-target-feature" "+rcpc" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+sha2" "-target-feature" "+aes"
// CHECK-MCPU-A34: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc"
// CHECK-MCPU-APPLE-A13: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.4a" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+dotprod" "-target-feature" "+fullfp16" "-target-feature" "+ras" "-target-feature" "+lse" "-target-feature" "+rdm" "-target-feature" "+rcpc" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+fp16fml" "-target-feature" "+sm4" "-target-feature" "+sha3" "-target-feature" "+sha2" "-target-feature" "+aes"
// CHECK-MCPU-A35: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-A53: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-A57: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-A72: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-CORTEX-A73: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-M1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-M4: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+dotprod" "-target-feature" "+fullfp16"
// CHECK-MCPU-KRYO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-THUNDERX2T99: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-A64FX: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+fullfp16" "-target-feature" "+ras" "-target-feature" "+lse" "-target-feature" "+rdm" "-target-feature" "+sve" "-target-feature" "+sha2"
// CHECK-MCPU-CARMEL: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+fullfp16" "-target-feature" "+ras" "-target-feature" "+lse" "-target-feature" "+rdm" "-target-feature" "+sha2" "-target-feature" "+aes"

// RUN: %clang -target x86_64-apple-macosx -arch arm64 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH-ARM64 %s
// CHECK-ARCH-ARM64: "-target-cpu" "apple-a7" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crypto" "-target-feature" "+zcm" "-target-feature" "+zcz"

// RUN: %clang -target x86_64-apple-macosx -arch arm64_32 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH-ARM64_32 %s
// CHECK-ARCH-ARM64_32: "-target-cpu" "apple-s4" "-target-feature" "+v8.3a" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+fullfp16" "-target-feature" "+ras" "-target-feature" "+lse" "-target-feature" "+rdm" "-target-feature" "+rcpc" "-target-feature" "+zcm" "-target-feature" "+zcz" "-target-feature" "+sha2" "-target-feature" "+aes"

// RUN: %clang -target aarch64 -march=armv8-a+fp+simd+crc+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-1 %s
// RUN: %clang -target aarch64 -march=armv8-a+nofp+nosimd+nocrc+nocrypto+fp+simd+crc+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-1 %s
// RUN: %clang -target aarch64 -march=armv8-a+nofp+nosimd+nocrc+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-2 %s
// RUN: %clang -target aarch64 -march=armv8-a+fp+simd+crc+crypto+nofp+nosimd+nocrc+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-2 %s
// RUN: %clang -target aarch64 -march=armv8-a+nosimd -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-3 %s
// CHECK-MARCH-1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MARCH-2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "-fp-armv8" "-target-feature" "-neon" "-target-feature" "-crc" "-target-feature" "-crypto"
// CHECK-MARCH-3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "-neon"

// Check +sm4:
//
// RUN: %clang -target aarch64 -march=armv8.2a+sm4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-SM4 %s
// CHECK-SM4:  "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "+sm4"
//
// Check +sha3:
//
// RUN: %clang -target aarch64 -march=armv8.2a+sha3 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-SHA3 %s
// CHECK-SHA3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "+sha3"
//
// Check +sha2:
//
// RUN: %clang -target aarch64 -march=armv8.3a+sha2 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-SHA2 %s
// CHECK-SHA2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.{{.}}a" "-target-feature" "+sha2"
//
// Check +aes:
//
// RUN: %clang -target aarch64 -march=armv8.3a+aes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-AES %s
// CHECK-AES: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.{{.}}a" "-target-feature" "+aes"
//
// Check -sm4:
//
// RUN: %clang -target aarch64 -march=armv8.2a+noSM4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SM4 %s
// CHECK-NO-SM4:  "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "-sm4"
//
// Check -sha3:
//
// RUN: %clang -target aarch64 -march=armv8.2a+noSHA3 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SHA3 %s
// CHECK-NO-SHA3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "-sha3"
//
// Check -sha2:
//
// RUN: %clang -target aarch64 -march=armv8.2a+noSHA2 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-SHA2 %s
// CHECK-NO-SHA2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "-sha2"
//
// Check -aes:
//
// RUN: %clang -target aarch64 -march=armv8.2a+noAES -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NO-AES %s
// CHECK-NO-AES: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.2a" "-target-feature" "-aes"
//
//
// Arch <= ARMv8.3:  crypto = sha2 + aes
// -------------------------------------
//
// Check +crypto:
//
// RUN: %clang -target aarch64 -march=armv8a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO83 %s
// RUN: %clang -target aarch64 -march=armv8.1a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO83 %s
// RUN: %clang -target aarch64 -march=armv8.2a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO83 %s
// RUN: %clang -target aarch64 -march=armv8.3a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO83 %s
// RUN: %clang -target aarch64 -march=armv8a+crypto+nocrypto+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO83 %s
// CHECK-CRYPTO83: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+crypto" "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -crypto:
//
// RUN: %clang -target aarch64 -march=armv8a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO8A %s
// RUN: %clang -target aarch64 -march=armv8.1a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO81 %s
// RUN: %clang -target aarch64 -march=armv8.2a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO82 %s
// RUN: %clang -target aarch64 -march=armv8.3a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO82 %s
// RUN: %clang -target aarch64 -march=armv8.3a+nocrypto+crypto+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO82 %s

// CHECK-NOCRYPTO8A: "-target-feature" "+neon" "-target-feature" "-crypto" "-target-feature" "-sha2" "-target-feature" "-aes" "-target-abi" "aapcs"
// CHECK-NOCRYPTO81: "-target-feature" "+neon" "-target-feature" "+v8.1a" "-target-feature" "-crypto" "-target-feature" "-sha2" "-target-feature" "-aes" "-target-abi" "aapcs"
// CHECK-NOCRYPTO82: "-target-feature" "+neon" "-target-feature" "+v8.{{.}}a" "-target-feature" "-crypto" "-target-feature" "-sha2" "-target-feature" "-aes" "-target-feature" "-sm4" "-target-feature" "-sha3" "-target-abi" "aapcs"
//
// Check +crypto -sha2 -aes:
//
// RUN: %clang -target aarch64 -march=armv8.1a+crypto+nosha2+noaes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO83-NOSHA2-NOAES %s
// CHECK-CRYPTO83-NOSHA2-NOAES-NOT: "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -crypto +sha2 +aes:
//
// RUN: %clang -target aarch64 -march=armv8.1a+nocrypto+sha2+aes -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO83-SHA2-AES %s
// CHECK-NOCRYPTO83-SHA2-AES: "-target-feature" "+sha2" "-target-feature" "+aes"
//
//
// Arch >= ARMv8.4:  crypto = sm4 + sha3 + sha2 + aes
// --------------------------------------------------
//
// Check +crypto:
//
// RUN: %clang -target aarch64 -march=armv8.4a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO84 %s
// CHECK-CRYPTO84: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.4a" "-target-feature" "+crypto" "-target-feature" "+sm4" "-target-feature" "+sha3" "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check -crypto:
//
// RUN: %clang -target aarch64 -march=armv8.4a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-NOCRYPTO84 %s
// CHECK-NOCRYPTO84-NOT: "-target-feature" "+crypto" "-target-feature" "+sm4" "-target-feature" "+sha3" "-target-feature" "+sha2" "-target-feature" "+aes"
//
// Check +crypto -sm4 -sha3:
//
// RUN: %clang -target aarch64 -march=armv8.4a+crypto+sm4+nosm4+sha3+nosha3 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-CRYPTO84-NOSMSHA %s
// CHECK-CRYPTO84-NOSMSHA: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.4a" "-target-feature" "+crypto" "-target-feature" "-sm4" "-target-feature" "-sha3" "-target-feature" "+sha2" "-target-feature" "+aes"
//
//
// RUN: %clang -target aarch64 -mcpu=cyclone+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-1 %s
// RUN: %clang -target aarch64 -mcpu=cyclone+crypto+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-1 %s
// RUN: %clang -target aarch64 -mcpu=generic+crc -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-2 %s
// RUN: %clang -target aarch64 -mcpu=generic+nocrc+crc -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-2 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a53+nosimd -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-3 %s
// ================== Check whether -mcpu accepts mixed-case features.
// RUN: %clang -target aarch64 -mcpu=cyclone+NOCRYPTO -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-1 %s
// RUN: %clang -target aarch64 -mcpu=cyclone+CRYPTO+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-1 %s
// RUN: %clang -target aarch64 -mcpu=generic+Crc -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-2 %s
// RUN: %clang -target aarch64 -mcpu=GENERIC+nocrc+CRC -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-2 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a53+noSIMD -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-3 %s
// CHECK-MCPU-1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "-crypto" "-target-feature" "+zcm" "-target-feature" "+zcz"
// CHECK-MCPU-2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc"
// CHECK-MCPU-3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "-neon"

// RUN: %clang -target aarch64 -mcpu=cyclone+nocrc+nocrypto -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-MARCH %s
// RUN: %clang -target aarch64 -march=armv8-a -mcpu=cyclone+nocrc+nocrypto  -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-MARCH %s
// CHECK-MCPU-MARCH: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+zcm" "-target-feature" "+zcz"

// RUN: %clang -target aarch64 -mcpu=cortex-a53 -mtune=cyclone -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-MTUNE %s
// RUN: %clang -target aarch64 -mtune=cyclone -mcpu=cortex-a53  -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-MTUNE %s
// ================== Check whether -mtune accepts mixed-case features.
// RUN: %clang -target aarch64 -mcpu=cortex-a53 -mtune=CYCLONE -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-MTUNE %s
// RUN: %clang -target aarch64 -mtune=CyclonE -mcpu=cortex-a53  -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-MTUNE %s
// CHECK-MCPU-MTUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto" "-target-feature" "+zcm" "-target-feature" "+zcz"

// RUN: %clang -target aarch64 -mcpu=generic+neon -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-NEON %s
// RUN: %clang -target aarch64 -mcpu=generic+noneon -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-NEON %s
// RUN: %clang -target aarch64 -march=armv8-a+neon -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-NEON %s
// RUN: %clang -target aarch64 -march=armv8-a+noneon -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-ERROR-NEON %s
// CHECK-ERROR-NEON: error: [no]neon is not accepted as modifier, please use [no]simd instead

// RUN: %clang -target aarch64 -march=armv8.1a+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-FEATURE-1 %s
// RUN: %clang -target aarch64 -march=armv8.1a+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-FEATURE-2 %s
// RUN: %clang -target aarch64 -march=armv8.1a+nosimd -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-FEATURE-3 %s
// ================== Check whether -march accepts mixed-case features.
// RUN: %clang -target aarch64 -march=ARMV8.1A+CRYPTO -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-FEATURE-1 %s
// RUN: %clang -target aarch64 -march=Armv8.1a+NOcrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-FEATURE-2 %s
// RUN: %clang -target aarch64 -march=armv8.1a+noSIMD -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V81A-FEATURE-3 %s
// CHECK-V81A-FEATURE-1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+v8.1a" "-target-feature" "+crypto"
// CHECK-V81A-FEATURE-2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+v8.1a" "-target-feature" "-crypto"
// CHECK-V81A-FEATURE-3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+v8.1a" "-target-feature" "-neon"

// ================== Check Memory Tagging Extensions (MTE).
// RUN: %clang -target arm64-none-linux-gnu -march=armv8.5-a+memtag -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-MEMTAG %s
// CHECK-MEMTAG: __ARM_FEATURE_MEMORY_TAGGING 1

// ================== Check Pointer Authentication Extension (PAuth).
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8.5-a -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=none -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=bti -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=standard -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret+b-key -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH-BKEY %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret+leaf -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH-ALL %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret+leaf+b-key -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-PAUTH-BKEY-ALL %s
// CHECK-PAUTH-OFF-NOT:  __ARM_FEATURE_PAC_DEFAULT
// CHECK-PAUTH:          #define __ARM_FEATURE_PAC_DEFAULT 1
// CHECK-PAUTH-BKEY:     #define __ARM_FEATURE_PAC_DEFAULT 2
// CHECK-PAUTH-ALL:      #define __ARM_FEATURE_PAC_DEFAULT 5
// CHECK-PAUTH-BKEY-ALL: #define __ARM_FEATURE_PAC_DEFAULT 6

// ================== Check Branch Target Identification (BTI).
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8.5-a -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=none -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret+leaf -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret+b-key -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret+leaf+b-key -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI-OFF %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=standard -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=bti -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI %s
// RUN: %clang -target arm64-none-linux-gnu -march=armv8-a -mbranch-protection=pac-ret+bti -x c -E -dM %s -o - | FileCheck -check-prefix=CHECK-BTI %s
// CHECK-BTI-OFF-NOT: __ARM_FEATURE_BTI_DEFAULT
// CHECK-BTI:         #define __ARM_FEATURE_BTI_DEFAULT 1

// ================== Check BFloat16 Extensions.
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.6-a+bf16 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-BFLOAT %s
// CHECK-BFLOAT: __ARM_BF16_FORMAT_ALTERNATIVE 1
// CHECK-BFLOAT: __ARM_FEATURE_BF16 1
// CHECK-BFLOAT: __ARM_FEATURE_BF16_VECTOR_ARITHMETIC 1

// ================== Check sve-vector-bits flag.
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-a+sve -msve-vector-bits=128 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-SVE-VECTOR-BITS-128 %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-a+sve -msve-vector-bits=256 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-SVE-VECTOR-BITS-256 %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-a+sve -msve-vector-bits=512 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-SVE-VECTOR-BITS-512 %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-a+sve -msve-vector-bits=1024 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-SVE-VECTOR-BITS-1024 %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-a+sve -msve-vector-bits=2048 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-SVE-VECTOR-BITS-2048 %s
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8-a+sve -msve-vector-bits=2048 -x c -E -dM %s -o - 2>&1 | FileCheck -check-prefix=CHECK-SVE-VECTOR-BITS-2048 %s
// NOTE: The __ARM_FEATURE_SVE_BITS feature macro is experimental until the
// feature is complete.
// CHECK-SVE-VECTOR-BITS-128: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 128
// CHECK-SVE-VECTOR-BITS-256: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 256
// CHECK-SVE-VECTOR-BITS-512: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 512
// CHECK-SVE-VECTOR-BITS-1024: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 1024
// CHECK-SVE-VECTOR-BITS-2048: __ARM_FEATURE_SVE_BITS_EXPERIMENTAL 2048
