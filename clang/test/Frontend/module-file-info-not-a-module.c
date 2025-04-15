// RUN: not %clang_cc1 -module-file-info %s 2>&1 | FileCheck %s

// CHECK: fatal error: file '{{.*}}module-file-info-not-a-module.c' is not a module file
