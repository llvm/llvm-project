// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --product-name=TypedefChain -triple arm64-apple-macosx -x c-header %s -o %t/typedefchain-c.symbols.json -verify
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --product-name=TypedefChain -triple arm64-apple-macosx -x c++-header %s -o %t/typedefchain-cxx.symbols.json -verify

// RUN: FileCheck %s --input-file %t/typedefchain-c.symbols.json --check-prefix MYSTRUCT
// RUN: FileCheck %s --input-file %t/typedefchain-cxx.symbols.json --check-prefix MYSTRUCT
typedef struct _MyStruct { } MyStruct;

// MYSTRUCT-LABEL: "!testLabel": "c:@S@_MyStruct"
// MYSTRUCT:      "accessLevel": "public",
// MYSTRUCT:      "declarationFragments": [
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "keyword",
// MYSTRUCT-NEXT:     "spelling": "typedef"
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "text",
// MYSTRUCT-NEXT:     "spelling": " "
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "keyword",
// MYSTRUCT-NEXT:     "spelling": "struct"
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "text",
// MYSTRUCT-NEXT:     "spelling": " "
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "identifier",
// MYSTRUCT-NEXT:     "spelling": "_MyStruct"
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "text",
// MYSTRUCT-NEXT:     "spelling": " { ... } "
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "identifier",
// MYSTRUCT-NEXT:     "spelling": "MyStruct"
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "text",
// MYSTRUCT-NEXT:     "spelling": ";"
// MYSTRUCT-NEXT:   }
// MYSTRUCT-NEXT: ],
// MYSTRUCT:      "kind": {
// MYSTRUCT-NEXT:   "displayName": "Structure",
// MYSTRUCT-NEXT:   "identifier": "c{{(\+\+)?}}.struct"
// MYSTRUCT:           "names": {
// MYSTRUCT-NEXT:        "navigator": [
// MYSTRUCT-NEXT:          {
// MYSTRUCT-NEXT:            "kind": "identifier",
// MYSTRUCT-NEXT:            "spelling": "MyStruct"
// MYSTRUCT-NEXT:          }
// MYSTRUCT-NEXT:        ],
// MYSTRUCT-NEXT:        "subHeading": [
// MYSTRUCT-NEXT:          {
// MYSTRUCT-NEXT:            "kind": "identifier",
// MYSTRUCT-NEXT:            "spelling": "MyStruct"
// MYSTRUCT-NEXT:          }
// MYSTRUCT-NEXT:        ],
// MYSTRUCT-NEXT:        "title": "MyStruct"
// MYSTRUCT-NEXT:      },
// MYSTRUCT:      "pathComponents": [
// MYSTRUCT-NEXT:    "MyStruct"
// MYSTRUCT-NEXT:  ]

// expected-no-diagnostics
