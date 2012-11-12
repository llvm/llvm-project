// RUN: env QA_OVERRIDE_GCC3_OPTIONS="#+-Os +-Oz +-O +-O3 +-Oignore +a +b +c xb Xa Omagic ^-ccc-print-options  " %clang x -O2 b -O3 2>&1 | FileCheck %s
// CHECK-NOT: ###
// CHECK: Option 0 - Name: "-ccc-print-options", Values: {}
// CHECK-NEXT: Option 1 - Name: "<input>", Values: {"x"}
// CHECK-NEXT: Option 2 - Name: "-O", Values: {"ignore"}
// CHECK-NEXT: Option 3 - Name: "-O", Values: {"magic"}
