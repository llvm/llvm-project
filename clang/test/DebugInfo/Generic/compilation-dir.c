// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: cp %s rel.c
// RUN: %clang_cc1 -fdebug-compilation-dir /nonsense -emit-llvm -debug-info-kind=limited rel.c -o - | FileCheck -check-prefix=CHECK-NONSENSE %s
// RUN: %clang_cc1 -fdebug-compilation-dir=/nonsense -emit-llvm -debug-info-kind=limited rel.c -o - | FileCheck -check-prefix=CHECK-NONSENSE %s
// CHECK-NONSENSE: nonsense

// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck -check-prefix=CHECK-DIR %s
// CHECK-DIR: Generic

/// Test path remapping.
// RUN: %clang_cc1 -fdebug-compilation-dir=%S -main-file-name %s -emit-llvm -debug-info-kind=limited %s -o - | FileCheck -check-prefix=CHECK-ABS %s
// CHECK-ABS: DIFile(filename: "{{.*}}compilation-dir.c", directory: "{{.*}}Generic")

// RUN: %clang_cc1 -main-file-name %s -emit-llvm -debug-info-kind=limited %s -o - | FileCheck -check-prefix=CHECK-NOMAP %s
// CHECK-NOMAP: DIFile(filename: "{{.*}}compilation-dir.c", directory: "")

