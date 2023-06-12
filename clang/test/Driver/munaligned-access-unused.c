/// Check -m[no-]unaligned-access and -m[no-]strict-align are warned unused on a target that does not support them.

// RUN: %clang --target=x86_64 -munaligned-access -fsyntax-only %s -### 2>&1 | FileCheck %s -DOPTION=unaligned-access
// RUN: %clang --target=x86_64 -mno-unaligned-access -fsyntax-only %s -### 2>&1 | FileCheck %s -DOPTION=no-unaligned-access
// RUN: %clang --target=x86_64 -mstrict-align -fsyntax-only %s -### 2>&1 | FileCheck %s -DOPTION=strict-align
// RUN: %clang --target=x86_64 -mno-strict-align -fsyntax-only %s -### 2>&1 | FileCheck %s -DOPTION=no-strict-align

// CHECK: error: unsupported option '-m{{(no-)?}}unaligned-access' for target '{{.*}}'
