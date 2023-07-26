// RUN: not %clang --target=loongarch64 -mtune=invalidcpu -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang --target=loongarch64 -mtune=generic -fsyntax-only %s 2>&1 | FileCheck %s
// RUN: not %clang --target=loongarch64 -mtune=generic-la64 -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: error: unknown target CPU '{{.*}}'
// CHECK-NEXT: note: valid target CPU values are: {{.*}}
