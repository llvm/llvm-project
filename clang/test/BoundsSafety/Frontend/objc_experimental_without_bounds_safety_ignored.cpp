

// RUN: %clang -cc1 -fbounds-attributes-objc-experimental -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: warning: -fbounds-attributes-objc-experimental without -fbounds-attributes is ignored
