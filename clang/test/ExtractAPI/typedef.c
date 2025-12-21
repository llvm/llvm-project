// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing -fblocks \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %s -o %t/output.symbols.json -verify

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix MYINT
typedef int MyInt;
// MYINT-LABEL: "!testLabel": "c:typedef.c@T@MyInt"
// MYINT: "accessLevel": "public",
// MYINT:      "declarationFragments": [
// MYINT-NEXT:   {
// MYINT-NEXT:     "kind": "keyword",
// MYINT-NEXT:     "spelling": "typedef"
// MYINT-NEXT:   },
// MYINT-NEXT:   {
// MYINT-NEXT:     "kind": "text",
// MYINT-NEXT:     "spelling": " "
// MYINT-NEXT:   },
// MYINT-NEXT:   {
// MYINT-NEXT:     "kind": "typeIdentifier",
// MYINT-NEXT:     "preciseIdentifier": "c:I",
// MYINT-NEXT:     "spelling": "int"
// MYINT-NEXT:   },
// MYINT-NEXT:   {
// MYINT-NEXT:     "kind": "text",
// MYINT-NEXT:     "spelling": " "
// MYINT-NEXT:   },
// MYINT-NEXT:   {
// MYINT-NEXT:     "kind": "identifier",
// MYINT-NEXT:     "spelling": "MyInt"
// MYINT-NEXT:   },
// MYINT-NEXT:   {
// MYINT-NEXT:     "kind": "text",
// MYINT-NEXT:     "spelling": ";"
// MYINT-NEXT:   }
// MYINT-NEXT: ],
// MYINT:      "kind": {
// MYINT-NEXT:   "displayName": "Type Alias",
// MYINT-NEXT:   "identifier": "objective-c.typealias"
// MYINT-NEXT: },
// MYINT: "title": "MyInt"
// MYINT:      "pathComponents": [
// MYINT-NEXT:   "MyInt"
// MYINT-NEXT: ],
// MYINT: "type": "c:I"

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix BARPTR
typedef struct Bar *BarPtr;
// BARPTR-LABEL: "!testLabel": "c:typedef.c@T@BarPtr"
// BARPTR: "accessLevel": "public",
// BARPTR:      "declarationFragments": [
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "keyword",
// BARPTR-NEXT:     "spelling": "typedef"
// BARPTR-NEXT:   },
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "text",
// BARPTR-NEXT:     "spelling": " "
// BARPTR-NEXT:   },
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "keyword",
// BARPTR-NEXT:     "spelling": "struct"
// BARPTR-NEXT:   },
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "text",
// BARPTR-NEXT:     "spelling": " "
// BARPTR-NEXT:   },
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "typeIdentifier",
// BARPTR-NEXT:     "preciseIdentifier": "c:@S@Bar",
// BARPTR-NEXT:     "spelling": "Bar"
// BARPTR-NEXT:   },
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "text",
// BARPTR-NEXT:     "spelling": " * "
// BARPTR-NEXT:   },
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "identifier",
// BARPTR-NEXT:     "spelling": "BarPtr"
// BARPTR-NEXT:   },
// BARPTR-NEXT:   {
// BARPTR-NEXT:     "kind": "text",
// BARPTR-NEXT:     "spelling": ";"
// BARPTR-NEXT:   }
// BARPTR-NEXT: ],
// BARPTR: "type": "c:*$@S@Bar"

// RUN: FileCheck %s --input-file %t/output.symbols.json
void foo(BarPtr value);

void baz(BarPtr *value);
// CHECK-NOT: struct Bar *

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix BLOCKPTR
typedef int (^CustomType)(const unsigned int *, unsigned long);
void bar(CustomType block);

// BLOCKPTR-LABEL: "!testLabel": "c:@F@bar",
// BLOCKPTR:           "declarationFragments": [
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "typeIdentifier",
// BLOCKPTR-NEXT:          "preciseIdentifier": "c:v",
// BLOCKPTR-NEXT:          "spelling": "void"
// BLOCKPTR-NEXT:        },
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "text",
// BLOCKPTR-NEXT:          "spelling": " "
// BLOCKPTR-NEXT:        },
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "identifier",
// BLOCKPTR-NEXT:          "spelling": "bar"
// BLOCKPTR-NEXT:        },
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "text",
// BLOCKPTR-NEXT:          "spelling": "("
// BLOCKPTR-NEXT:        },
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "typeIdentifier",
// BLOCKPTR-NEXT:          "preciseIdentifier": "c:typedef.c@T@CustomType",
// BLOCKPTR-NEXT:          "spelling": "CustomType"
// BLOCKPTR-NEXT:        },
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "text",
// BLOCKPTR-NEXT:          "spelling": " "
// BLOCKPTR-NEXT:        },
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "internalParam",
// BLOCKPTR-NEXT:          "spelling": "block"
// BLOCKPTR-NEXT:        },
// BLOCKPTR-NEXT:        {
// BLOCKPTR-NEXT:          "kind": "text",
// BLOCKPTR-NEXT:          "spelling": ");"
// BLOCKPTR-NEXT:        }
// BLOCKPTR-NEXT:      ],
// BLOCKPTR-NEXT:      "functionSignature": {
// BLOCKPTR-NEXT:        "parameters": [
// BLOCKPTR-NEXT:          {
// BLOCKPTR-NEXT:            "declarationFragments": [
// BLOCKPTR-NEXT:              {
// BLOCKPTR-NEXT:                "kind": "typeIdentifier",
// BLOCKPTR-NEXT:                "preciseIdentifier": "c:typedef.c@T@CustomType",
// BLOCKPTR-NEXT:                "spelling": "CustomType"
// BLOCKPTR-NEXT:              },
// BLOCKPTR-NEXT:              {
// BLOCKPTR-NEXT:                "kind": "text",
// BLOCKPTR-NEXT:                "spelling": " "
// BLOCKPTR-NEXT:              },
// BLOCKPTR-NEXT:              {
// BLOCKPTR-NEXT:                "kind": "internalParam",
// BLOCKPTR-NEXT:                "spelling": "block"
// BLOCKPTR-NEXT:              }
// BLOCKPTR-NEXT:            ],
// BLOCKPTR-NEXT:            "name": "block"
// BLOCKPTR-NEXT:          }
// BLOCKPTR-NEXT:        ],
// BLOCKPTR-NEXT:        "returns": [
// BLOCKPTR-NEXT:          {
// BLOCKPTR-NEXT:            "kind": "typeIdentifier",
// BLOCKPTR-NEXT:            "preciseIdentifier": "c:v",
// BLOCKPTR-NEXT:            "spelling": "void"
// BLOCKPTR-NEXT:          }
// BLOCKPTR-NEXT:        ]
// BLOCKPTR-NEXT:      },
// BLOCKPTR:           "identifier": {
// BLOCKPTR-NEXT:        "interfaceLanguage": "objective-c",
// BLOCKPTR-NEXT:        "precise": "c:@F@bar"
// BLOCKPTR-NEXT:      },
// BLOCKPTR:           "kind": {
// BLOCKPTR-NEXT:        "displayName": "Function",
// BLOCKPTR-NEXT:        "identifier": "objective-c.func"
// BLOCKPTR-NEXT:      },

// expected-no-diagnostics
