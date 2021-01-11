/// -fno-pic code defaults to -fdirect-access-external-data.
// RUN: %clang -### -c -target x86_64 %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target x86_64 %s -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target x86_64 %s -fdirect-access-external-data -fno-direct-access-external-data 2>&1 | FileCheck %s --check-prefix=INDIRECT

/// -fpie/-fpic code defaults to -fdirect-access-external-data.
// RUN: %clang -### -c -target x86_64 %s -fpie 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target x86_64 %s -fpie -fno-direct-access-external-data -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DIRECT
// RUN: %clang -### -c -target aarch64 %s -fpic 2>&1 | FileCheck %s --check-prefix=DEFAULT
// RUN: %clang -### -c -target aarch64 %s -fpic -fdirect-access-external-data 2>&1 | FileCheck %s --check-prefix=DIRECT

/// -m[no-]pie-copy-relocations are aliases for compatibility.
// RUN: %clang -### -c -target riscv64 %s -mno-pie-copy-relocations 2>&1 | FileCheck %s --check-prefix=INDIRECT
// RUN: %clang -### -c -target riscv64 %s -fpic -mpie-copy-relocations 2>&1 | FileCheck %s --check-prefix=DIRECT

// DEFAULT-NOT: direct-access-external-data"
// DIRECT:      "-fdirect-access-external-data"
// INDIRECT:    "-fno-direct-access-external-data"
