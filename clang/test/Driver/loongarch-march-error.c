// RUN: not %clang --target=loongarch64 -march=loongarch -fsyntax-only %s 2>&1 | \
// RUN:   FileCheck -DCPU=loongarch %s
// RUN: not %clang --target=loongarch64 -march=LA464 -fsyntax-only %s 2>&1 | \
// RUN:   FileCheck -DCPU=LA464 %s

// CHECK: error: unknown target CPU '[[CPU]]'
// CHECK-NEXT: note: valid target CPU values are: {{.*}}
