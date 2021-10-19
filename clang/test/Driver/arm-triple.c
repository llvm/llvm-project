// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabi \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clang -print-effective-triple \
// RUN:   --target=armeb-none-eabi -mlittle-endian \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabihf -march=armv4t -mfloat-abi=softfp \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT: armv4t-none-unknown-eabi

// RUN: %clang -print-effective-triple \
// RUN:   --target=armeb-none-eabi \
// RUN:   | FileCheck %s --check-prefix=CHECK-EB
// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabi -mbig-endian \
// RUN:   | FileCheck %s --check-prefix=CHECK-EB
// CHECK-EB: armebv4t-none-unknown-eabi

// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabihf -march=armv4t \
// RUN:   | FileCheck %s --check-prefix=CHECK-HF
// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabi -mfloat-abi=hard \
// RUN:   | FileCheck %s --check-prefix=CHECK-HF
// CHECK-HF: armv4t-none-unknown-eabihf

// RUN: %clang -print-effective-triple \
// RUN:   --target=armeb-none-eabihf -march=armv4t \
// RUN:   | FileCheck %s --check-prefix=CHECK-EB-HF
// RUN: %clang -print-effective-triple \
// RUN:   --target=armeb-none-eabi -mfloat-abi=hard \
// RUN:   | FileCheck %s --check-prefix=CHECK-EB-HF
// RUN: %clang -print-effective-triple -march=armv4t \
// RUN:   --target=arm-none-eabihf -mbig-endian \
// RUN:   | FileCheck %s --check-prefix=CHECK-EB-HF
// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabi -mbig-endian -mfloat-abi=hard \
// RUN:   | FileCheck %s --check-prefix=CHECK-EB-HF
// CHECK-EB-HF: armebv4t-none-unknown-eabihf

// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabi -march=armv8m.main -mbig-endian -mfloat-abi=hard \
// RUN:   | FileCheck %s --check-prefix=CHECK-V8M-EB-HF
// RUN: %clang -print-effective-triple \
// RUN:   --target=arm-none-eabi -mcpu=cortex-m33 -mbig-endian -mfloat-abi=hard \
// RUN:   | FileCheck %s --check-prefix=CHECK-V8M-EB-HF
// CHECK-V8M-EB-HF: thumbebv8m.main-none-unknown-eabihf
