// RUN: %clang -### -Xlinker -Bsymbolic -emit-interface-stubs 2>&1 | FileCheck %s
// CHECK: Bsymbolic
// CHECK-NOT: Bsymbolic
