// RUN: %clang -ccc-print-options -working-directory "C:\Test" input 2>&1 | FileCheck %s
// CHECK: Option 0 - Name: "-ccc-print-options", Values: {}
// CHECK: Option 1 - Name: "-working-directory", Values: {"C:\Test"}
// CHECK: Option 2 - Name: "<input>", Values: {"input"}
