/// -fno-pic code defaults to -fdirect-access-external-data.
// RUN: %clang -### -c -target x86_64 %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target x86_64 %s -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target x86_64 %s -fdirect-access-external-data -fno-direct-access-external-data 2>&1 | FileCheck %s --check-prefix=INDIRECT

/// -fno-pic code defaults to -fno-direct-access-external-data.
// RUN: %clang -### -c -target loongarch64 %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target loongarch64 %s -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: %clang -### -c -target loongarch64 %s -fdirect-access-external-data -fno-direct-access-external-data 2>&1 | FileCheck %s --check-prefix=DEFAULT

/// -fpie/-fpic code defaults to -fdirect-access-external-data.
// RUN: %clang -### -c -target x86_64 %s -fpie 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target x86_64 %s -fpie -fno-direct-access-external-data -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: %clang -### -c -target aarch64 %s -fpic 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target aarch64 %s -fpic -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DIRECT

/// -fpie/-fpic code defaults to -fno-direct-access-external-data.
// RUN: %clang -### -c -target loongarch64 %s -fpie 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target loongarch64 %s -fpie -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: %clang -### -c -target loongarch64 %s -fpie -fdirect-access-external-data -fno-direct-access-external-data 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target loongarch64 %s -fpic 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target loongarch64 %s -fpic -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: %clang -### -c -target loongarch64 %s -fpic -fdirect-access-external-data -fno-direct-access-external-data 2>&1 | FileCheck %s --check-prefix=DEFAULT

// DEFAULT-NOT: direct-access-external-data"
// DIRECT:      "-fdirect-access-external-data"
// INDIRECT:    "-fno-direct-access-external-data"
