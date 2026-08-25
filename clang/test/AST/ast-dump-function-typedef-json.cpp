// RUN: %clang_cc1 -std=c++11 -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

typedef double (*TestFunctionPointer)(int first, const char *second);
using TestFunctionAlias = void(long third, bool fourth);

// CHECK:      "kind": "TypedefDecl"
// CHECK:      "name": "TestFunctionPointer"
// CHECK:      "kind": "ParmVarDecl"
// CHECK:      "name": "first"
// CHECK:      "type": {
// CHECK-NEXT:   "qualType": "int"
// CHECK:      "kind": "ParmVarDecl"
// CHECK:      "name": "second"
// CHECK:      "type": {
// CHECK-NEXT:   "qualType": "const char *"

// CHECK:      "kind": "TypeAliasDecl"
// CHECK:      "name": "TestFunctionAlias"
// CHECK:      "kind": "ParmVarDecl"
// CHECK:      "name": "third"
// CHECK:      "type": {
// CHECK-NEXT:   "qualType": "long"
// CHECK:      "kind": "ParmVarDecl"
// CHECK:      "name": "fourth"
// CHECK:      "type": {
// CHECK-NEXT:   "qualType": "bool"
