// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: cp %s rel.c
// RUN: %clang_cc1 -fdebug-compilation-dir /nonsense -emit-llvm -debug-info-kind=limited rel.c -o - | FileCheck -check-prefix=CHECK-NONSENSE %s
// RUN: %clang_cc1 -fdebug-compilation-dir=/nonsense -emit-llvm -debug-info-kind=limited rel.c -o - | FileCheck -check-prefix=CHECK-NONSENSE %s
// CHECK-NONSENSE: nonsense

// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck -check-prefix=CHECK-DIR %s
// CHECK-DIR: CodeGen

/// Test path remapping.
// RUN: %clang_cc1 -fdebug-compilation-dir %S -main-file-name %s -emit-llvm -debug-info-kind=limited %s -o - | FileCheck -check-prefix=CHECK-ABS %s -DPREFIX=%S
// CHECK-ABS: DIFile(filename: "[[PREFIX]]/debug-info-compilation-dir.c", directory: "[[PREFIX]]")

// RUN: %clang_cc1 -fno-compilation-dir -main-file-name %s -emit-llvm -debug-info-kind=limited %s -o - | FileCheck -check-prefix=CHECK-NOMAP %s -DPREFIX=%S
// CHECK-NOMAP: DIFile(filename: "[[PREFIX]]/debug-info-compilation-dir.c", directory: "")

