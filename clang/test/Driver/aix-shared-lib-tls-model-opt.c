// RUN: %clang -target powerpc64-unknown-aix -S -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK-AIX,CHECK-AIX-OFF %s
// RUN: %clang -target powerpc-unknown-aix -S -emit-llvm %s -o - | FileCheck --check-prefixes=CHECK-AIX,CHECK-AIX-OFF %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-LINUX %s
// RUN: %clang -target powerpc64-unknown-linux-gnu -S -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-LINUX %s

// RUN: %clang -target powerpc64-unknown-aix -maix-shared-lib-tls-model-opt -S -emit-llvm \
// RUN:    %s -o - | FileCheck %s --check-prefixes=CHECK-AIX,CHECK-AIX-ON

// RUN: not %clang -target powerpc-unknown-aix -maix-shared-lib-tls-model-opt \
// RUN:    -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED-TARGET %s
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -maix-shared-lib-tls-model-opt \
// RUN:    -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED-TARGET %s
// RUN: not %clang -target powerpc64-unknown-linux-gnu -maix-shared-lib-tls-model-opt \
// RUN:    -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-UNSUPPORTED-TARGET %s

int test(void) {
  return 0;
}

// CHECK-AIX: test() #0 {
// CHECK-AIX: attributes #0 = {
// CHECK-AIX-OFF-SAME: -aix-shared-lib-tls-model-opt
// CHECK-AIX-ON-SAME: +aix-shared-lib-tls-model-opt

// CHECK-LINUX-NOT: {{[-+]aix-shared-lib-tls-model-opt}}

// CHECK-UNSUPPORTED-TARGET: option '-maix-shared-lib-tls-model-opt' cannot be specified on this target
