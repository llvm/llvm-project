// RUN: %clang_cc1 %s -O900 -o /dev/null 2>&1 | FileCheck %s

// RUN: %clang_cc1 %s -O8 -o /dev/null 2>&1 | FileCheck %s

// CHECK: warning: optimization level '-O{{.*}}' is not supported; using '-O3' instead
