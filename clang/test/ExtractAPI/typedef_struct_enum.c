// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -x c-header %s -triple arm64-apple-macos -o %t/output.symbols.json -verify

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix TEST
typedef struct Test {
} Test;
// TEST-LABEL: "!testLabel": "c:@S@Test"
// TEST:      "declarationFragments": [
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "keyword",
// TEST-NEXT:     "spelling": "typedef"
// TEST-NEXT:   },
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "text",
// TEST-NEXT:     "spelling": " "
// TEST-NEXT:   },
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "keyword",
// TEST-NEXT:     "spelling": "struct"
// TEST-NEXT:   },
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "text",
// TEST-NEXT:     "spelling": " "
// TEST-NEXT:   },
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "identifier",
// TEST-NEXT:     "spelling": "Test"
// TEST-NEXT:   },
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "text",
// TEST-NEXT:     "spelling": " { ... } "
// TEST-NEXT:   },
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "identifier",
// TEST-NEXT:     "spelling": "Test"
// TEST-NEXT:   },
// TEST-NEXT:   {
// TEST-NEXT:     "kind": "text",
// TEST-NEXT:     "spelling": ";"
// TEST-NEXT:   }
// TEST-NEXT: ],
// TEST: "displayName": "Structure",
// TEST: "title": "Test"

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix TEST2
typedef enum Test2 {
  simple
} Test2;

// TEST2-LABEL: "!testLabel": "c:@E@Test2"
// TEST2:      "declarationFragments": [
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "keyword",
// TEST2-NEXT:     "spelling": "typedef"
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "text",
// TEST2-NEXT:     "spelling": " "
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "keyword",
// TEST2-NEXT:     "spelling": "enum"
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "text",
// TEST2-NEXT:     "spelling": " "
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "identifier",
// TEST2-NEXT:     "spelling": "Test2"
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "text",
// TEST2-NEXT:     "spelling": ": "
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "typeIdentifier",
// TEST2-NEXT:     "preciseIdentifier": "c:i",
// TEST2-NEXT:     "spelling": "unsigned int"
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "text",
// TEST2-NEXT:     "spelling": " { ... } "
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "identifier",
// TEST2-NEXT:     "spelling": "Test2"
// TEST2-NEXT:   },
// TEST2-NEXT:   {
// TEST2-NEXT:     "kind": "text",
// TEST2-NEXT:     "spelling": ";"
// TEST2-NEXT:   }
// TEST2-NEXT: ],
// TEST2: "displayName": "Enumeration",
// TEST2: "title": "Test2"

struct Foo;

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix TYPEDEF
typedef struct Foo TypedefedFoo;
// TYPEDEF-LABEL: "!testLabel": "c:typedef_struct_enum.c@T@TypedefedFoo"
// TYPEDEF:      "declarationFragments": [
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "keyword",
// TYPEDEF-NEXT:     "spelling": "typedef"
// TYPEDEF-NEXT:   },
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "text",
// TYPEDEF-NEXT:     "spelling": " "
// TYPEDEF-NEXT:   },
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "keyword",
// TYPEDEF-NEXT:     "spelling": "struct"
// TYPEDEF-NEXT:   },
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "text",
// TYPEDEF-NEXT:     "spelling": " "
// TYPEDEF-NEXT:   },
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "typeIdentifier",
// TYPEDEF-NEXT:     "preciseIdentifier": "c:@S@Foo",
// TYPEDEF-NEXT:     "spelling": "Foo"
// TYPEDEF-NEXT:   },
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "text",
// TYPEDEF-NEXT:     "spelling": " "
// TYPEDEF-NEXT:   },
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "identifier",
// TYPEDEF-NEXT:     "spelling": "TypedefedFoo"
// TYPEDEF-NEXT:   },
// TYPEDEF-NEXT:   {
// TYPEDEF-NEXT:     "kind": "text",
// TYPEDEF-NEXT:     "spelling": ";"
// TYPEDEF-NEXT:   }
// TYPEDEF-NEXT: ],
// TYPEDEF: "displayName": "Type Alias",
// TYPEDEF: "title": "TypedefedFoo"
// TYPEDEF: "type": "c:@S@Foo"

struct Foo {
    int bar;
};

// expected-no-diagnostics
