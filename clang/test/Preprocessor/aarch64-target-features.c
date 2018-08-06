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

// RUN: %clang -target aarch64-none-linux-gnu -march=armv8.2a+dotprod -x c -E -dM %s -o - | FileCheck --check-prefix=CHECK-DOTPROD %s
// CHECK-DOTPROD: __ARM_FEATURE_DOTPROD 1

// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2a+fp16 -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-VECTOR-SCALAR %s
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FP 0xE
// CHECK-FULLFP16-VECTOR-SCALAR: #define __ARM_FP16_FORMAT_IEEE 1

// RUN: %clang -target aarch64-none-linux-gnueabi -march=armv8.2a+fp16+nosimd -x c -E -dM %s -o - | FileCheck -match-full-lines --check-prefix=CHECK-FULLFP16-SCALAR %s
// CHECK-FULLFP16-SCALAR:       #define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 1
// CHECK-FULLFP16-SCALAR-NOT:   #define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 1
// CHECK-FULLFP16-SCALAR:       #define __ARM_FP 0xE
// CHECK-FULLFP16-SCALAR:       #define __ARM_FP16_FORMAT_IEEE 1

// ================== Check whether -mtune accepts mixed-case features.
// RUN: %clang -target aarch64 -mtune=CYCLONE -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MTUNE-CYCLONE %s
// CHECK-MTUNE-CYCLONE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+zcm" "-target-feature" "+zcz"

// RUN: %clang -target aarch64 -mcpu=cyclone -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-CYCLONE %s
// RUN: %clang -target aarch64 -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A35 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A53 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A57 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-A72 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-CORTEX-A73 %s
// RUN: %clang -target aarch64 -mcpu=exynos-m1 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-M1 %s
// RUN: %clang -target aarch64 -mcpu=exynos-m2 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-M1 %s
// RUN: %clang -target aarch64 -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-M1 %s
// RUN: %clang -target aarch64 -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-M1 %s
// RUN: %clang -target aarch64 -mcpu=kryo -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-KRYO %s
// RUN: %clang -target aarch64 -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MCPU-THUNDERX2T99 %s
// CHECK-MCPU-CYCLONE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crypto" "-target-feature" "+zcm" "-target-feature" "+zcz"
// CHECK-MCPU-A35: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-A53: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-A57: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-A72: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-CORTEX-A73: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-M1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-KRYO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MCPU-THUNDERX2T99: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"

// RUN: %clang -target x86_64-apple-macosx -arch arm64 -### -c %s 2>&1 | FileCheck --check-prefix=CHECK-ARCH-ARM64 %s
// CHECK-ARCH-ARM64: "-target-cpu" "cyclone" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crypto" "-target-feature" "+zcm" "-target-feature" "+zcz"

// RUN: %clang -target aarch64 -march=armv8-a+fp+simd+crc+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-1 %s
// RUN: %clang -target aarch64 -march=armv8-a+nofp+nosimd+nocrc+nocrypto+fp+simd+crc+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-1 %s
// RUN: %clang -target aarch64 -march=armv8-a+nofp+nosimd+nocrc+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-2 %s
// RUN: %clang -target aarch64 -march=armv8-a+fp+simd+crc+crypto+nofp+nosimd+nocrc+nocrypto -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-2 %s
// RUN: %clang -target aarch64 -march=armv8-a+nosimd -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-MARCH-3 %s
// CHECK-MARCH-1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+crc" "-target-feature" "+crypto"
// CHECK-MARCH-2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "-fp-armv8" "-target-feature" "-neon" "-target-feature" "-crc" "-target-feature" "-crypto"
// CHECK-MARCH-3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "-neon"

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

