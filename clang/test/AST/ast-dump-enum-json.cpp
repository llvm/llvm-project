// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++17 -ast-dump=json %s | FileCheck %s

enum {
  One,
  Two
};

// CHECK: "kind": "EnumDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 3
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 3
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 6
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 4
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "One",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "(anonymous enum at {{.*}}:3:1)"
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 5
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 5
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 5
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Two",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "(anonymous enum at {{.*}}:3:1)"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

enum E {
  Three,
  Four
};

// CHECK: "kind": "EnumDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 6,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 80
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 80
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 83
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "E",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 81
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 81
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 81
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Three",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "E"
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 82
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 82
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 82
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Four",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "E"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

enum F : short {
  Five,
  Six
};

// CHECK: "kind": "EnumDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 6,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 158
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 158
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 161
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "F",
// CHECK-NEXT: "fixedUnderlyingType": {
// CHECK-NEXT: "qualType": "short"
// CHECK-NEXT: },
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 159
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 159
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 159
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Five",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "F"
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 160
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 160
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 160
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Six",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "F"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

enum struct G {
  Seven,
  Eight
};

// CHECK: "kind": "EnumDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 13,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 239
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 239
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 242
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "G",
// CHECK-NEXT: "fixedUnderlyingType": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "scopedEnumTag": "struct",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 240
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 240
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 240
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Seven",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "G"
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 241
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 241
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 241
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Eight",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "G"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

enum class H {
  Nine,
  Ten
};

// CHECK: "kind": "EnumDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 321
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 321
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 324
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "H",
// CHECK-NEXT: "fixedUnderlyingType": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "scopedEnumTag": "class",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 322
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 322
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 322
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Nine",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "H"
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 323
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 323
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 323
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Ten",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "H"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: },

enum class I : int {
  Eleven,
  Twelve
};

// CHECK: "kind": "EnumDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 12,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 403
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 403
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 1,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 406
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "I",
// CHECK-NEXT: "fixedUnderlyingType": {
// CHECK-NEXT: "qualType": "int"
// CHECK-NEXT: },
// CHECK-NEXT: "scopedEnumTag": "class",
// CHECK-NEXT: "inner": [
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 404
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 404
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 404
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Eleven",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "I"
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: {
// CHECK-NEXT: "id": "0x{{.*}}",
// CHECK-NEXT: "kind": "EnumConstantDecl",
// CHECK-NEXT: "loc": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 405
// CHECK-NEXT: },
// CHECK-NEXT: "range": {
// CHECK-NEXT: "begin": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 405
// CHECK-NEXT: },
// CHECK-NEXT: "end": {
// CHECK-NEXT: "col": 3,
// CHECK-NEXT: "file": "{{.*}}",
// CHECK-NEXT: "line": 405
// CHECK-NEXT: }
// CHECK-NEXT: },
// CHECK-NEXT: "name": "Twelve",
// CHECK-NEXT: "type": {
// CHECK-NEXT: "qualType": "I"
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: ]
// CHECK-NEXT: }
// CHECK-NEXT: ]
