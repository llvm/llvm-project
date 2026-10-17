// Atomic operations are only lock-free on 32-bit SPARC CPUs with a
// compare-and-swap: V9 ones, and the LEON and Myriad 2 ones with CASA.

// RUN: %clang --target=sparc-unknown-none-elf -x c -E -dM %s -o - \
// RUN:   | FileCheck %s --check-prefix=NOCAS
// RUN: %clang --target=sparcel-unknown-none-elf -x c -E -dM %s -o - \
// RUN:   | FileCheck %s --check-prefix=NOCAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=v8 -x c -E -dM %s -o - \
// RUN:   | FileCheck %s --check-prefix=NOCAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=leon2 -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=NOCAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=ut699 -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=NOCAS

// RUN: %clang --target=sparc-unknown-none-elf -mcpu=leon3 -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=gr712rc -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=leon4 -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=gr740 -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=leon5 -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=ma2100 -x c -E -dM %s \
// RUN:   -o - | FileCheck %s --check-prefix=CAS
// RUN: %clang --target=sparc-unknown-none-elf -mcpu=v9 -x c -E -dM %s -o - \
// RUN:   | FileCheck %s --check-prefix=CAS
// RUN: %clang --target=sparc-unknown-linux-gnu -x c -E -dM %s -o - \
// RUN:   | FileCheck %s --check-prefix=CAS

// NOCAS: #define __GCC_ATOMIC_BOOL_LOCK_FREE 1
// NOCAS: #define __GCC_ATOMIC_CHAR_LOCK_FREE 1
// NOCAS: #define __GCC_ATOMIC_INT_LOCK_FREE 1
// NOCAS: #define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// NOCAS: #define __GCC_ATOMIC_LONG_LOCK_FREE 1
// NOCAS: #define __GCC_ATOMIC_POINTER_LOCK_FREE 1
// NOCAS: #define __GCC_ATOMIC_SHORT_LOCK_FREE 1

// CAS: #define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// CAS: #define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// CAS: #define __GCC_ATOMIC_INT_LOCK_FREE 2
// CAS: #define __GCC_ATOMIC_LLONG_LOCK_FREE 1
// CAS: #define __GCC_ATOMIC_LONG_LOCK_FREE 2
// CAS: #define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// CAS: #define __GCC_ATOMIC_SHORT_LOCK_FREE 2
