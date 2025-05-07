

// RUN: %clang -cc1 -fexperimental-bounds-safety-objc -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: warning: -fexperimental-bounds-safety-objc without -fbounds-safety is ignored
