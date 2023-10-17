// RUN: %clang -print-multi-flags-experimental --target=aarch64-linux -fc++-abi=itanium -fsanitize=address | FileCheck --check-prefix=CHECK-LINUX %s
// CHECK-LINUX: --target=aarch64-unknown-linux

// RUN: %clang -print-multi-flags-experimental --target=aarch64-fuchsia -fsanitize=hwaddress | FileCheck --check-prefix=CHECK-FUCHSIA %s
// CHECK-FUCHSIA: --target=aarch64-unknown-fuchsia

// RUN: %clang -print-multi-flags-experimental --target=arm-none-eabi -mfloat-abi=soft -fno-exceptions -fno-rtti | FileCheck --check-prefix=CHECK-ARMV4T %s
// CHECK-ARMV4T: --target=armv4t-none-unknown-eabi
// CHECK-ARMV4T: -mfloat-abi=soft
// CHECK-ARMV4T: -mfpu=none

// RUN: %clang -print-multi-flags-experimental --target=armv7em-none-eabi -mfloat-abi=softfp | FileCheck --check-prefix=CHECK-SOFTFP %s
// CHECK-SOFTFP: --target=thumbv7em-none-unknown-eabi
// CHECK-SOFTFP: -mfloat-abi=softfp
// CHECK-SOFTFP: -mfpu=fpv4-sp-d16

// RUN: %clang -print-multi-flags-experimental --target=arm-none-eabihf -march=armv7em -mfpu=fpv5-d16 | FileCheck --check-prefix=CHECK-HARD %s
// CHECK-HARD: --target=thumbv7em-none-unknown-eabihf
// CHECK-HARD: -mfloat-abi=hard
// CHECK-HARD: -mfpu=fpv5-d16

// RUN: %clang -print-multi-flags-experimental --target=arm-none-eabi -mfloat-abi=soft -march=armv8-m.main+nofp | FileCheck --check-prefix=CHECK-V8MMAIN-NOFP %s
// CHECK-V8MMAIN-NOFP: --target=thumbv8m.main-none-unknown-eabi
// CHECK-V8MMAIN-NOFP: -mfloat-abi=soft
// CHECK-V8MMAIN-NOFP: -mfpu=none

// RUN: %clang -print-multi-flags-experimental --target=arm-none-eabi -mfloat-abi=hard -march=armv8.1m.main+mve.fp | FileCheck --check-prefix=CHECK-MVE %s
// CHECK-MVE: --target=thumbv8.1m.main-none-unknown-eabihf
// CHECK-MVE: -march=thumbv8.1m.main{{.*}}+mve{{.*}}+mve.fp{{.*}}
// CHECK-MVE: -mfloat-abi=hard
// CHECK-MVE: -mfpu=fp-armv8-fullfp16-sp-d16

// RUN: %clang -print-multi-flags-experimental --target=arm-none-eabi -march=armv8.1m.main+mve+nofp | FileCheck --check-prefix=CHECK-MVENOFP %s
// CHECK-MVENOFP: -march=thumbv8.1m.main{{.*}}+mve{{.*}}
// CHECK-MVENOFP-NOT: -march=thumbv8.1m.main{{.*}}+mve.fp{{.*}}
// CHECK-MVENOFP: -mfpu=none

// RUN: %clang -print-multi-flags-experimental --target=aarch64-none-elf -march=armv8-a+lse | FileCheck --check-prefix=CHECK-LSE %s
// CHECK-LSE: -march=aarch64{{.*}}+lse{{.*}}

// RUN: %clang -print-multi-flags-experimental --target=aarch64-none-elf -march=armv8.5-a+sve+sve2 | FileCheck --check-prefix=CHECK-SVE2 %s
// RUN: %clang -print-multi-flags-experimental --target=aarch64-none-elf -march=armv9-a            | FileCheck --check-prefix=CHECK-SVE2 %s
// CHECK-SVE2: --target=aarch64-none-unknown-elf
// CHECK-SVE2: -march=aarch64{{.*}}+simd{{.*}}+sve{{.*}}+sve2{{.*}}
