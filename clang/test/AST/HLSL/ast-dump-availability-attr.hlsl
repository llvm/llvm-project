// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump=json %s | FileCheck %s

// Test AvailabilityAttr fields in JSON AST dump

__attribute__((availability(shadermodel, introduced=6.0, deprecated=6.3, obsoleted=6.5, replacement="new_func", environment=compute)))
void availability_all(void);

// CHECK: "kind": "FunctionDecl",
// CHECK: "name": "availability_all",
// CHECK: "kind": "AvailabilityAttr",
// CHECK: "platform": "shadermodel",
// CHECK: "introduced": "6.0",
// CHECK: "deprecated": "6.3",
// CHECK: "obsoleted": "6.5",
// CHECK: "replacement": "new_func",
// CHECK: "environment": "compute"
