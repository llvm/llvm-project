/// Treat -falign-loops=0 as not specifying the option.
// RUN: %clang -### -falign-loops=0 %s 2>&1 | FileCheck %s --check-prefix=CHECK_NO
// RUN: %clang -### -falign-loops=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK_1
// RUN: %clang -### -falign-loops=4 %s 2>&1 | FileCheck %s --check-prefix=CHECK_4
/// Only powers of 2 are supported for now.
// RUN: not %clang -### -falign-loops=5 %s 2>&1 | FileCheck %s --check-prefix=CHECK_5
// RUN: %clang -### -falign-loops=65536 %s 2>&1 | FileCheck %s --check-prefix=CHECK_65536
// RUN: not %clang -### -falign-loops=65537 %s 2>&1 | FileCheck %s --check-prefix=CHECK_65537
// RUN: not %clang -### -falign-loops=a %s 2>&1 | FileCheck %s --check-prefix=CHECK_ERR_A

// CHECK_NO-NOT: "-falign-loops=
// CHECK_1: "-falign-loops=1"
// CHECK_4: "-falign-loops=4"
// CHECK_5: error: alignment is not a power of 2 in '-falign-loops=5'
// CHECK_65536: "-falign-loops=65536"
// CHECK_65537: error: invalid integral value '65537' in '-falign-loops=65537'
// CHECK_ERR_A: error: invalid integral value 'a' in '-falign-loops=a'
