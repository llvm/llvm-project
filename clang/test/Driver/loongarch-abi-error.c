// RUN: not %clang --target=loongarch32-unknown-elf %s -fsyntax-only -mabi=lp64s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LA32-LP64S %s
// RUN: not %clang --target=loongarch32-unknown-elf %s -fsyntax-only -mabi=lp64f 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LA32-LP64F %s
// RUN: not %clang --target=loongarch32-unknown-elf %s -fsyntax-only -mabi=lp64d 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LA32-LP64D %s

// RUN: not %clang --target=loongarch64-unknown-elf %s -fsyntax-only -mabi=ilp32s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LA64-ILP32S %s
// RUN: not %clang --target=loongarch64-unknown-elf %s -fsyntax-only -mabi=ilp32f 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LA64-ILP32F %s
// RUN: not %clang --target=loongarch64-unknown-elf %s -fsyntax-only -mabi=ilp32d 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LA64-ILP32D %s

// CHECK-LA32-LP64S: error: unknown target ABI 'lp64s'
// CHECK-LA32-LP64F: error: unknown target ABI 'lp64f'
// CHECK-LA32-LP64D: error: unknown target ABI 'lp64d'

// CHECK-LA64-ILP32S: error: unknown target ABI 'ilp32s'
// CHECK-LA64-ILP32F: error: unknown target ABI 'ilp32f'
// CHECK-LA64-ILP32D: error: unknown target ABI 'ilp32d'
