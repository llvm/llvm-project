// REQUIRES: arm-registered-target

// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o -               | FileCheck %s --check-prefix=NO-ROPI --check-prefix=NO-RWPI
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -fropi        | FileCheck %s --check-prefix=ROPI    --check-prefix=NO-RWPI
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -frwpi        | FileCheck %s --check-prefix=NO-ROPI --check-prefix=RWPI
// RUN: %clang -target armv8--none-eabi   -x c -E -dM %s -o - -fropi -frwpi | FileCheck %s --check-prefix=ROPI    --check-prefix=RWPI

// Pre-defined macros for position-independence modes

// NO-ROPI-NOT: #define __APCS_ROPI
// ROPI: #define __ARM_ROPI

// NO-RWPI-NOT: #define __APCS_RWPI
// RWPI: #define __ARM_RWPI
