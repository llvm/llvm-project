// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++17 -ast-dump=json %s | FileCheck %s

void func(int val) {
  if (val)
    ;

// CHECK: "kind": "IfStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 5
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "IntegralToBoolean",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "LValueToRValue",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclRefExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "lvalue",
// CHECK-NEXT: "referencedDecl": {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ParmVarDecl",
// CHECK-NEXT: "name": "val",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 5
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 5
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

  if (val)
    ;
  else
    ;

// CHECK: "kind": "IfStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 114
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 117
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "hasElse": true,
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 114
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 114
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "IntegralToBoolean",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 114
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 114
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "LValueToRValue",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclRefExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 114
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 114
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "lvalue"
// CHECK-NEXT: "referencedDecl": {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ParmVarDecl",
// CHECK-NEXT: "name": "val",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 115
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 115
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 117
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 117
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

  if (val)
    ;
  else if (val)
    ;
  else
    ;

// CHECK: "kind": "IfStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 243
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 248
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "hasElse": true,
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 243
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 243
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "IntegralToBoolean",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 243
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 243
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "LValueToRValue",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclRefExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 243
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 243
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "lvalue"
// CHECK-NEXT: "referencedDecl": {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ParmVarDecl",
// CHECK-NEXT: "name": "val",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 244
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 244
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "IfStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 8,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 245
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 248
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "hasElse": true,
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 245
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 245
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "IntegralToBoolean",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 245
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 245
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "LValueToRValue",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclRefExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 245
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 245
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "lvalue"
// CHECK-NEXT: "referencedDecl": {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ParmVarDecl",
// CHECK-NEXT: "name": "val",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 246
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 246
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 248
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 248
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

  if constexpr(10 == 10)
    ;

// CHECK: "kind": "IfStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 484
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "isConstexpr": true,
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ConstantExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 16,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 22,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "BinaryOperator",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 16,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 22,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "opcode": "==",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "IntegerLiteral",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 16,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 16,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "value": "10"
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "IntegerLiteral",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 22,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 22,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 483
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "value": "10"
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 484
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 484
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

  if (int i = 12)
    ;

// CHECK: "kind": "IfStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 608
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "hasVar": true,
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 15,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "VarDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 15,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "isUsed": true,
// CHECK-NEXT: "name": "i",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "init": "c",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "IntegerLiteral",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 15,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 15,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "value": "12"
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "IntegralToBoolean",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "LValueToRValue",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclRefExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 607
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "lvalue"
// CHECK-NEXT: "referencedDecl": {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "VarDecl",
// CHECK-NEXT: "name": "i",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 608
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 608
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

  if (int i = 12; i)
    ;

// CHECK: "kind": "IfStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 787
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "hasInit": true,
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 17,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "VarDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 11,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 7,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 15,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "isUsed": true,
// CHECK-NEXT: "name": "i",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "init": "c",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "IntegerLiteral",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 15,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 15,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "value": "12"
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 19,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 19,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "bool"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "IntegralToBoolean",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "ImplicitCastExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 19,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 19,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "rvalue",
// CHECK-NEXT: "castKind": "LValueToRValue",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "DeclRefExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 19,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 19,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 786
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "lvalue"
// CHECK-NEXT: "referencedDecl": {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "VarDecl",
// CHECK-NEXT: "name": "i",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "NullStmt",
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 787
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 5,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 787
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
}
