// RUN: clang-tidy -checks='-*,misc-unused-parameters' -dump-config %s -- 2>/dev/null | FileCheck %s --check-prefix=CHECK
// RUN: clang-tidy -checks='-*' -dump-config %s -- 2>/dev/null | FileCheck %s --check-prefix=CHECK-DISABLED

// CHECK: CheckOptions:
// CHECK-NEXT:   misc-unused-parameters.IgnoreVirtual: 'false'
// CHECK-NEXT:   misc-unused-parameters.StrictMode: 'false'
// CHECK-NEXT: SystemHeaders:   false

// CHECK-DISABLED: CheckOptions:    {}
// CHECK-DISABLED-NEXT: SystemHeaders:   false

int main() { return 0; }
