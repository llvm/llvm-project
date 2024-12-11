
// RUN: %clang_cc1 -fbounds-safety-adoption-mode  -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety-adoption-mode -fbounds-safety -fno-bounds-safety  -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -fno-bounds-safety -fbounds-safety-adoption-mode  -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-adoption-mode -fno-bounds-safety -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: warning: -fbounds-safety-adoption-mode without -fbounds-safety is ignored
