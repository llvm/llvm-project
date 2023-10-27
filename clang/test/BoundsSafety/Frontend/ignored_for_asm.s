
// RUN: %clang -fbounds-safety-experimental -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety-experimental -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: warning: '-fbounds-safety' is ignored for assembly
