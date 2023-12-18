// RUN: %clang -S -### -fopenacc %s 2>&1 | FileCheck %s --check-prefix=CHECK-DRIVER
// CHECK-DRIVER: "-cc1" {{.*}} "-fopenacc"

// RUN: %clang -S -### -fopenacc -fexperimental-openacc-macro-override=202211 %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACRO-OVERRIDE
// RUN: %clang -S -### -fopenacc -fexperimental-openacc-macro-override 202211 %s 2>&1 | FileCheck %s --check-prefix=CHECK-MACRO-OVERRIDE
// CHECK-MACRO-OVERRIDE: "-cc1"{{.*}} "-fexperimental-openacc-macro-override" "202211"

// RUN: not %clang -S -fopenacc -fexperimental-openacc-macro-override=202211L %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -S -fopenacc -fexperimental-openacc-macro-override 202211L %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -S -fopenacc -fexperimental-openacc-macro-override=L202211 %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -S -fopenacc -fexperimental-openacc-macro-override L202211 %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -S -fopenacc -fexperimental-openacc-macro-override=2022L11 %s 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not %clang -S -fopenacc -fexperimental-openacc-macro-override 2022L11 %s 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: error: the clang compiler does not support
