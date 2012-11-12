// RUN: env CCC_ADD_ARGS="-ccc-echo,-ccc-print-options,,-v" %clang -### 2>&1 | FileCheck %s
// CHECK: Option 0 - Name: "-ccc-echo", Values: {}
// CHECK: Option 1 - Name: "-ccc-print-options", Values: {}
// CHECK: Option 2 - Name: "-v", Values: {}
// CHECK: Option 3 - Name: "-###", Values: {}
