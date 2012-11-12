// RUN: %clang -ccc-print-options \
// RUN:  --save-temps --undefine-macro=FOO --undefine-macro FOO \
// RUN: --param=FOO --output=FOO 2> %t
// RUN: FileCheck --check-prefix=CHECK-OPTIONS < %t %s

// CHECK-OPTIONS: Option 0 - Name: "-ccc-print-options", Values: {}
// CHECK-OPTIONS: Option 1 - Name: "-save-temps", Values: {}
// CHECK-OPTIONS: Option 2 - Name: "-U", Values: {"FOO"}
// CHECK-OPTIONS: Option 3 - Name: "-U", Values: {"FOO"}
// CHECK-OPTIONS: Option 4 - Name: "--param", Values: {"FOO"}
// CHECK-OPTIONS: Option 5 - Name: "-o", Values: {"FOO"}
