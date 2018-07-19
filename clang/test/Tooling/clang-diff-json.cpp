// RUN: clang-diff -ast-dump-json %s -- \
// RUN: | %python -c 'import json, sys; json.dump(json.loads(sys.stdin.read()), sys.stdout, sort_keys=True, indent=2)' \
// RUN: | FileCheck %s

// CHECK: "begin": 299,
// CHECK: "type": "FieldDecl",
// CHECK: "end": 319,
// CHECK: "type": "CXXRecordDecl",
class A {
  int x;
};

// CHECK: "children": [
// CHECK-NEXT: {
// CHECK-NEXT: "begin":
// CHECK-NEXT: "children": []
// CHECK-NEXT: "end":
// CHECK-NEXT: "id":
// CHECK-NEXT: "type": "CharacterLiteral"
// CHECK-NEXT: }
// CHECK: ]
// CHECK: "type": "VarDecl",
char nl = '\n';

// CHECK: "value": "abc \n\t\u0000\u001f"
char s[] = "abc \n\t\0\x1f";

