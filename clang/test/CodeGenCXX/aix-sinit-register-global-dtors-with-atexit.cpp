// RUN: not %clang_cc1 -triple powerpc-ibm-aix-xcoff -S -emit-llvm -x c++ \
// RUN:     -fregister-global-dtors-with-atexit < %s 2>&1 | \
// RUN:   FileCheck %s

// RUN: not %clang_cc1 -triple powerpc64-ibm-aix-xcoff -S -emit-llvm -x c++ \
// RUN:     -fregister-global-dtors-with-atexit < %s 2>&1 | \
// RUN:   FileCheck %s

struct T {
  T();
  ~T();
} t;

// CHECK: error in backend: register global dtors with atexit() is not supported yet
