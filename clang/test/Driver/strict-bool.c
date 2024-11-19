// RUN: %clang -### %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang -### -fstrict-bool %s 2>&1 | FileCheck %s --check-prefix=CHECK-STRICT
// RUN: %clang -### -fno-strict-bool %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONSTRICT
// RUN: %clang -### -fno-strict-bool=truncate %s 2>&1 | FileCheck %s --check-prefix=CHECK-TRUNCATE
// RUN: %clang -### -fno-strict-bool=nonzero %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONZERO
// RUN: %clang -### -fstrict-bool -fno-strict-bool %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONSTRICT
// RUN: %clang -### -fno-strict-bool -fno-strict-bool=nonzero %s 2>&1 | FileCheck %s --check-prefix=CHECK-NONZERO
// RUN: %clang -### -fno-strict-bool=nonzero -fstrict-bool %s 2>&1 | FileCheck %s --check-prefix=CHECK-STRICT

// RUN: %clang -### -mkernel %s 2>&1 | FileCheck %s --check-prefix=CHECK-TRUNCATE
// RUN: %clang -### -fapple-kext %s 2>&1 | FileCheck %s --check-prefix=CHECK-TRUNCATE
// RUN: %clang -### -mkernel -fstrict-bool %s 2>&1 | FileCheck %s --check-prefix=CHECK-STRICT
// RUN: %clang -### -fstrict-bool -mkernel %s 2>&1 | FileCheck %s --check-prefix=CHECK-STRICT

// RUN: not %clang -### -fno-strict-bool=ow-ouch %s 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID

// CHECK-NONE-NOT: -load-bool-from-mem
// CHECK-STRICT: -load-bool-from-mem=strict
// CHECK-NONSTRICT: -load-bool-from-mem=nonstrict
// CHECK-TRUNCATE: -load-bool-from-mem=truncate
// CHECK-NONZERO: -load-bool-from-mem=nonzero
// CHECK-INVALID: invalid value 'ow-ouch' in '-fno-strict-bool=ow-ouch'
