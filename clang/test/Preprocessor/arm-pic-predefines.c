// REQUIRES: arm-registered-target

// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o -               | FileCheck %s --check-prefix=NO-ROPI --check-prefix=NO-RWPI --check-prefix=NO-STATIC-BASE
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -fropi        | FileCheck %s --check-prefix=ROPI    --check-prefix=NO-RWPI --check-prefix=NO-STATIC-BASE
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -frwpi        | FileCheck %s --check-prefix=NO-ROPI --check-prefix=RWPI --check-prefix=NO-STATIC-BASE
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -fropi -frwpi | FileCheck %s --check-prefix=ROPI    --check-prefix=RWPI --check-prefix=NO-STATIC-BASE
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -fPIC         | FileCheck %s --check-prefix=NO-STATIC-BASE
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -fPIC -mno-pic-data-is-text-relative | FileCheck %s --check-prefix=STATIC-BASE
// RUN: %clang -target thumbv8--none-eabi -x c -E -dM %s -o - -fPIE -mno-pic-data-is-text-relative | FileCheck %s --check-prefix=STATIC-BASE

// Pre-defined macros for position-independence modes

// NO-ROPI-NOT: #define __APCS_ROPI
// ROPI: #define __ARM_ROPI

// NO-RWPI-NOT: #define __APCS_RWPI
// RWPI: #define __ARM_RWPI

// NO-STATIC-BASE-NOT: #define __ARM_PIC_DATA_STATIC_BASE
// STATIC-BASE: #define __ARM_PIC_DATA_STATIC_BASE 1
