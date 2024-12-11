

// RUN: %clang_cc1 -fbounds-attributes-cxx-experimental -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: warning: -fbounds-attributes-cxx-experimental without -fbounds-attributes is ignored
