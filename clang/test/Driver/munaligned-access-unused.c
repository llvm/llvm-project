/// Check -m[no-]unaligned-access and -m[no-]strict-align are errored on a target that does not support them.

// RUN: not %clang --target=x86_64 -munaligned-access -fsyntax-only %s -### 2>&1 | FileCheck %s -DOPTION=unaligned-access
// RUN: not %clang --target=x86_64 -mno-unaligned-access -fsyntax-only %s -### 2>&1 | FileCheck %s -DOPTION=no-unaligned-access
// RUN: not %clang --target=x86_64 -mno-strict-align -mstrict-align -fsyntax-only %s -### 2>&1 | FileCheck %s --check-prefix=ALIGN

// CHECK: error: unsupported option '-m{{(no-)?}}unaligned-access' for target '{{.*}}'
// ALIGN: error: unsupported option '-mno-strict-align' for target '{{.*}}'
// ALIGN: error: unsupported option '-mstrict-align' for target '{{.*}}'
