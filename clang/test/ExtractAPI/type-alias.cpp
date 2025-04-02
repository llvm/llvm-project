// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --product-name=TypeAlias -triple arm64-apple-macosx -x c++-header %s -o %t/type-alias.symbols.json -verify

// RUN: FileCheck %s --input-file %t/type-alias.symbols.json --check-prefix MYALIAS
using MyAlias = int;
//MYALIAS-LABEL "!testLabel": "c:@MYALIAS"
//MYALIAS:       "accessLevel": "public",
//MYALIAS:       "declarationFragments": [
//MYALIAS-NEXT:    {
//MYALIAS-NEXT:      "kind": "keyword",
//MYALIAS-NEXT:      "spelling": "using"
//MYALIAS-NEXT:    },
//MYALIAS-NEXT:    {
//MYALIAS-NEXT:      "kind": "text",
//MYALIAS-NEXT:      "spelling": " "
//MYALIAS-NEXT:    },
//MYALIAS-NEXT:    {
//MYALIAS-NEXT:      "kind": "identifier",
//MYALIAS-NEXT:      "spelling": "MyAlias"
//MYALIAS-NEXT:    },
//MYALIAS-NEXT:    {
//MYALIAS-NEXT:      "kind": "text",
//MYALIAS-NEXT:      "spelling": " = "
//MYALIAS-NEXT:    },
//MYALIAS-NEXT:    {
//MYALIAS-NEXT:      "kind": "typeIdentifier",
//MYALIAS-NEXT:      "preciseIdentifier": "c:I",
//MYALIAS-NEXT:      "spelling": "int"
//MYALIAS-NEXT:    },
//MYALIAS-NEXT:    {
//MYALIAS-NEXT:      "kind": "text",
//MYALIAS-NEXT:      "spelling": ";"
//MYALIAS-NEXT:    }
//MYALIAS:       "kind": {
//MYALIAS-NEXT:      "displayName": "Type Alias",
//MYALIAS-NEXT:      "identifier": "c++.typealias"
//MYALIAS:       names": {
//MYALIAS-NEXT:    "navigator": [
//MYALIAS-NEXT:      {
//MYALIAS-NEXT:        "kind": "identifier",
//MYALIAS-NEXT:        "spelling": "MyAlias"
//MYALIAS-NEXT:      }
//MYALIAS-NEXT:    ],
//MYALIAS-NEXT:      "subHeading": [
//MYALIAS-NEXT:        {
//MYALIAS-NEXT:          "kind": "identifier",
//MYALIAS-NEXT:          "spelling": "MyAlias"
//MYALIAS-NEXT:        }
//MYALIAS-NEXT:    ],
//MYALIAS-NEXT:    "title": "MyAlias"
//MYALIAS:       "pathComponents": [
//MYALIAS-NEXT:    "MyAlias"
//MYALIAS-NEXT:  ]

// expected-no-diagnostics
