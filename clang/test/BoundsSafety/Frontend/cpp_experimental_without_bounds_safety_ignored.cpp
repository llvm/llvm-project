

// RUN: %clang_cc1 -fexperimental-bounds-safety-cxx -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: warning: -fexperimental-bounds-safety-cxx without -fbounds-safety is ignored
