// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing --product-name=Macros -triple arm64-apple-macosx \
// RUN:   -isystem %S -x objective-c-header %s -o %t/output.symbols.json

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix HELLO
#define HELLO 1
// HELLO-LABEL: "!testLabel": "c:@macro@HELLO"
// HELLO:      "accessLevel": "public",
// HELLO-NEXT: "declarationFragments": [
// HELLO-NEXT:   {
// HELLO-NEXT:     "kind": "keyword",
// HELLO-NEXT:     "spelling": "#define"
// HELLO-NEXT:   },
// HELLO-NEXT:   {
// HELLO-NEXT:     "kind": "text",
// HELLO-NEXT:     "spelling": " "
// HELLO-NEXT:   },
// HELLO-NEXT:   {
// HELLO-NEXT:     "kind": "identifier",
// HELLO-NEXT:     "spelling": "HELLO"
// HELLO-NEXT:   }
// HELLO-NEXT: ],
// HELLO:      "kind": {
// HELLO-NEXT:   "displayName": "Macro",
// HELLO-NEXT:   "identifier": "objective-c.macro"
// HELLO-NEXT: },
// HELLO-NEXT: "location": {
// HELLO-NEXT:   "position": {
// HELLO-NEXT:     "character": 8,
// HELLO-NEXT:     "line": [[# @LINE - 25]]
// HELLO-NEXT:   },
// HELLO-NEXT:   "uri": "file://{{.*}}/macros.c"
// HELLO-NEXT: },
// HELLO-NEXT: "names": {
// HELLO-NEXT:   "navigator": [
// HELLO-NEXT:     {
// HELLO-NEXT:       "kind": "identifier",
// HELLO-NEXT:       "spelling": "HELLO"
// HELLO-NEXT:     }
// HELLO-NEXT:   ],
// HELLO-NEXT:   "subHeading": [
// HELLO-NEXT:     {
// HELLO-NEXT:       "kind": "identifier",
// HELLO-NEXT:       "spelling": "HELLO"
// HELLO-NEXT:     }
// HELLO-NEXT:   ],
// HELLO-NEXT:   "title": "HELLO"
// HELLO-NEXT: },
// HELLO-NEXT: "pathComponents": [
// HELLO-NEXT:   "HELLO"
// HELLO-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix WORLD
#define WORLD 2
// WORLD-LABEL: "!testLabel": "c:@macro@WORLD"
// WORLD:      "accessLevel": "public",
// WORLD-NEXT: "declarationFragments": [
// WORLD-NEXT:   {
// WORLD-NEXT:     "kind": "keyword",
// WORLD-NEXT:     "spelling": "#define"
// WORLD-NEXT:   },
// WORLD-NEXT:   {
// WORLD-NEXT:     "kind": "text",
// WORLD-NEXT:     "spelling": " "
// WORLD-NEXT:   },
// WORLD-NEXT:   {
// WORLD-NEXT:     "kind": "identifier",
// WORLD-NEXT:     "spelling": "WORLD"
// WORLD-NEXT:   }
// WORLD-NEXT: ],
// WORLD:      "kind": {
// WORLD-NEXT:   "displayName": "Macro",
// WORLD-NEXT:   "identifier": "objective-c.macro"
// WORLD-NEXT: },
// WORLD-NEXT: "location": {
// WORLD-NEXT:   "position": {
// WORLD-NEXT:     "character": 8,
// WORLD-NEXT:     "line": [[# @LINE - 25]]
// WORLD-NEXT:   },
// WORLD-NEXT:   "uri": "file://{{.*}}/macros.c"
// WORLD-NEXT: },
// WORLD-NEXT: "names": {
// WORLD-NEXT:   "navigator": [
// WORLD-NEXT:     {
// WORLD-NEXT:       "kind": "identifier",
// WORLD-NEXT:       "spelling": "WORLD"
// WORLD-NEXT:     }
// WORLD-NEXT:   ],
// WORLD-NEXT:   "subHeading": [
// WORLD-NEXT:     {
// WORLD-NEXT:       "kind": "identifier",
// WORLD-NEXT:       "spelling": "WORLD"
// WORLD-NEXT:     }
// WORLD-NEXT:   ],
// WORLD-NEXT:   "title": "WORLD"
// WORLD-NEXT: },
// WORLD-NEXT: "pathComponents": [
// WORLD-NEXT:   "WORLD"
// WORLD-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix MACRO_FUN
#define MACRO_FUN(x) x x
// MACRO_FUN-LABEL: "!testLabel": "c:@macro@MACRO_FUN"
// MACRO_FUN-NEXT: "accessLevel": "public",
// MACRO_FUN-NEXT: "declarationFragments": [
// MACRO_FUN-NEXT:   {
// MACRO_FUN-NEXT:     "kind": "keyword",
// MACRO_FUN-NEXT:     "spelling": "#define"
// MACRO_FUN-NEXT:   },
// MACRO_FUN-NEXT:   {
// MACRO_FUN-NEXT:     "kind": "text",
// MACRO_FUN-NEXT:     "spelling": " "
// MACRO_FUN-NEXT:   },
// MACRO_FUN-NEXT:   {
// MACRO_FUN-NEXT:     "kind": "identifier",
// MACRO_FUN-NEXT:     "spelling": "MACRO_FUN"
// MACRO_FUN-NEXT:   },
// MACRO_FUN-NEXT:   {
// MACRO_FUN-NEXT:     "kind": "text",
// MACRO_FUN-NEXT:     "spelling": "("
// MACRO_FUN-NEXT:   },
// MACRO_FUN-NEXT:   {
// MACRO_FUN-NEXT:     "kind": "internalParam",
// MACRO_FUN-NEXT:     "spelling": "x"
// MACRO_FUN-NEXT:   },
// MACRO_FUN-NEXT:   {
// MACRO_FUN-NEXT:     "kind": "text",
// MACRO_FUN-NEXT:     "spelling": ")"
// MACRO_FUN-NEXT:   }
// MACRO_FUN-NEXT: ],
// MACRO_FUN:      "kind": {
// MACRO_FUN-NEXT:   "displayName": "Macro",
// MACRO_FUN-NEXT:   "identifier": "objective-c.macro"
// MACRO_FUN-NEXT: },
// MACRO_FUN-NEXT: "location": {
// MACRO_FUN-NEXT:   "position": {
// MACRO_FUN-NEXT:     "character": 8,
// MACRO_FUN-NEXT:     "line": [[# @LINE - 37]]
// MACRO_FUN-NEXT:   },
// MACRO_FUN-NEXT:   "uri": "file://{{.*}}/macros.c"
// MACRO_FUN-NEXT: },
// MACRO_FUN-NEXT: "names": {
// MACRO_FUN-NEXT:   "navigator": [
// MACRO_FUN-NEXT:     {
// MACRO_FUN-NEXT:       "kind": "identifier",
// MACRO_FUN-NEXT:       "spelling": "MACRO_FUN"
// MACRO_FUN-NEXT:     }
// MACRO_FUN-NEXT:   ],
// MACRO_FUN-NEXT:   "subHeading": [
// MACRO_FUN-NEXT:     {
// MACRO_FUN-NEXT:       "kind": "identifier",
// MACRO_FUN-NEXT:       "spelling": "MACRO_FUN"
// MACRO_FUN-NEXT:     }
// MACRO_FUN-NEXT:   ],
// MACRO_FUN-NEXT:   "title": "MACRO_FUN"
// MACRO_FUN-NEXT: },
// MACRO_FUN-NEXT: "pathComponents": [
// MACRO_FUN-NEXT:   "MACRO_FUN"
// MACRO_FUN-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix FUN
#define FUN(x, y, z) x + y + z
// FUN-LABEL: "!testLabel": "c:@macro@FUN"
// FUN-NEXT: "accessLevel": "public",
// FUN-NEXT: "declarationFragments": [
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "keyword",
// FUN-NEXT:     "spelling": "#define"
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "text",
// FUN-NEXT:     "spelling": " "
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "identifier",
// FUN-NEXT:     "spelling": "FUN"
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "text",
// FUN-NEXT:     "spelling": "("
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "internalParam",
// FUN-NEXT:     "spelling": "x"
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "text",
// FUN-NEXT:     "spelling": ", "
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "internalParam",
// FUN-NEXT:     "spelling": "y"
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "text",
// FUN-NEXT:     "spelling": ", "
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "internalParam",
// FUN-NEXT:     "spelling": "z"
// FUN-NEXT:   },
// FUN-NEXT:   {
// FUN-NEXT:     "kind": "text",
// FUN-NEXT:     "spelling": ")"
// FUN-NEXT:   }
// FUN-NEXT: ],
// FUN:      "kind": {
// FUN-NEXT:   "displayName": "Macro",
// FUN-NEXT:   "identifier": "objective-c.macro"
// FUN-NEXT: },
// FUN-NEXT: "location": {
// FUN-NEXT:   "position": {
// FUN-NEXT:     "character": 8,
// FUN-NEXT:     "line": [[# @LINE - 53]]
// FUN-NEXT:   },
// FUN-NEXT:   "uri": "file://{{.*}}/macros.c"
// FUN-NEXT: },
// FUN-NEXT: "names": {
// FUN-NEXT:   "navigator": [
// FUN-NEXT:     {
// FUN-NEXT:       "kind": "identifier",
// FUN-NEXT:       "spelling": "FUN"
// FUN-NEXT:     }
// FUN-NEXT:   ],
// FUN-NEXT:   "subHeading": [
// FUN-NEXT:     {
// FUN-NEXT:       "kind": "identifier",
// FUN-NEXT:       "spelling": "FUN"
// FUN-NEXT:     }
// FUN-NEXT:   ],
// FUN-NEXT:   "title": "FUN"
// FUN-NEXT: },
// FUN-NEXT: "pathComponents": [
// FUN-NEXT:   "FUN"
// FUN-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix FUNC99
#define FUNC99(x, ...)
// FUNC99-LABEL: "!testLabel": "c:@macro@FUNC99"
// FUNC99-NEXT: "accessLevel": "public",
// FUNC99-NEXT: "declarationFragments": [
// FUNC99-NEXT:   {
// FUNC99-NEXT:     "kind": "keyword",
// FUNC99-NEXT:     "spelling": "#define"
// FUNC99-NEXT:   },
// FUNC99-NEXT:   {
// FUNC99-NEXT:     "kind": "text",
// FUNC99-NEXT:     "spelling": " "
// FUNC99-NEXT:   },
// FUNC99-NEXT:   {
// FUNC99-NEXT:     "kind": "identifier",
// FUNC99-NEXT:     "spelling": "FUNC99"
// FUNC99-NEXT:   },
// FUNC99-NEXT:   {
// FUNC99-NEXT:     "kind": "text",
// FUNC99-NEXT:     "spelling": "("
// FUNC99-NEXT:   },
// FUNC99-NEXT:   {
// FUNC99-NEXT:     "kind": "internalParam",
// FUNC99-NEXT:     "spelling": "x"
// FUNC99-NEXT:   },
// FUNC99-NEXT:   {
// FUNC99-NEXT:     "kind": "text",
// FUNC99-NEXT:     "spelling": ", ...)"
// FUNC99-NEXT:   }
// FUNC99-NEXT: ],
// FUNC99:      "kind": {
// FUNC99-NEXT:   "displayName": "Macro",
// FUNC99-NEXT:   "identifier": "objective-c.macro"
// FUNC99-NEXT: },
// FUNC99-NEXT: "location": {
// FUNC99-NEXT:   "position": {
// FUNC99-NEXT:     "character": 8,
// FUNC99-NEXT:     "line": [[# @LINE - 37]]
// FUNC99-NEXT:   },
// FUNC99-NEXT:   "uri": "file://{{.*}}/macros.c"
// FUNC99-NEXT: },
// FUNC99-NEXT: "names": {
// FUNC99-NEXT:   "navigator": [
// FUNC99-NEXT:     {
// FUNC99-NEXT:       "kind": "identifier",
// FUNC99-NEXT:       "spelling": "FUNC99"
// FUNC99-NEXT:     }
// FUNC99-NEXT:   ],
// FUNC99-NEXT:   "subHeading": [
// FUNC99-NEXT:     {
// FUNC99-NEXT:       "kind": "identifier",
// FUNC99-NEXT:       "spelling": "FUNC99"
// FUNC99-NEXT:     }
// FUNC99-NEXT:   ],
// FUNC99-NEXT:   "title": "FUNC99"
// FUNC99-NEXT: },
// FUNC99-NEXT: "pathComponents": [
// FUNC99-NEXT:   "FUNC99"
// FUNC99-NEXT: ]

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix FUNGNU
#define FUNGNU(x...)
// FUNGNU-LABEL: "!testLabel": "c:@macro@FUNGNU"
// FUNGNU-NEXT: "accessLevel": "public",
// FUNGNU-NEXT: "declarationFragments": [
// FUNGNU-NEXT:   {
// FUNGNU-NEXT:     "kind": "keyword",
// FUNGNU-NEXT:     "spelling": "#define"
// FUNGNU-NEXT:   },
// FUNGNU-NEXT:   {
// FUNGNU-NEXT:     "kind": "text",
// FUNGNU-NEXT:     "spelling": " "
// FUNGNU-NEXT:   },
// FUNGNU-NEXT:   {
// FUNGNU-NEXT:     "kind": "identifier",
// FUNGNU-NEXT:     "spelling": "FUNGNU"
// FUNGNU-NEXT:   },
// FUNGNU-NEXT:   {
// FUNGNU-NEXT:     "kind": "text",
// FUNGNU-NEXT:     "spelling": "("
// FUNGNU-NEXT:   },
// FUNGNU-NEXT:   {
// FUNGNU-NEXT:     "kind": "internalParam",
// FUNGNU-NEXT:     "spelling": "x"
// FUNGNU-NEXT:   },
// FUNGNU-NEXT:   {
// FUNGNU-NEXT:     "kind": "text",
// FUNGNU-NEXT:     "spelling": "...)"
// FUNGNU-NEXT:   }
// FUNGNU-NEXT: ],
// FUNGNU:      "kind": {
// FUNGNU-NEXT:   "displayName": "Macro",
// FUNGNU-NEXT:   "identifier": "objective-c.macro"
// FUNGNU-NEXT: },
// FUNGNU-NEXT: "location": {
// FUNGNU-NEXT:   "position": {
// FUNGNU-NEXT:     "character": 8,
// FUNGNU-NEXT:     "line": [[# @LINE - 37]]
// FUNGNU-NEXT:   },
// FUNGNU-NEXT:   "uri": "file://{{.*}}/macros.c"
// FUNGNU-NEXT: },
// FUNGNU-NEXT: "names": {
// FUNGNU-NEXT:   "navigator": [
// FUNGNU-NEXT:     {
// FUNGNU-NEXT:       "kind": "identifier",
// FUNGNU-NEXT:       "spelling": "FUNGNU"
// FUNGNU-NEXT:     }
// FUNGNU-NEXT:   ],
// FUNGNU-NEXT:   "subHeading": [
// FUNGNU-NEXT:     {
// FUNGNU-NEXT:       "kind": "identifier",
// FUNGNU-NEXT:       "spelling": "FUNGNU"
// FUNGNU-NEXT:     }
// FUNGNU-NEXT:   ],
// FUNGNU-NEXT:   "title": "FUNGNU"
// FUNGNU-NEXT: },
// FUNGNU-NEXT: "pathComponents": [
// FUNGNU-NEXT:   "FUNGNU"
// FUNGNU-NEXT: ]

// expected-no-diagnostics

