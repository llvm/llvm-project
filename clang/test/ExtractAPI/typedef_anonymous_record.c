// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   --product-name=TypedefChain -triple arm64-apple-macosx -x c-header %s -o %t/typedefchain.symbols.json -verify

// RUN: Filecheck %s --input-file %t/typedefchain.symbols.json --check-prefix MYSTRUCT
typedef struct { } MyStruct;
// MYSTRUCT-LABEL: "!testLabel": "c:@SA@MyStruct"
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
// MYSTRUCT-NEXT:     "spelling": "MyStruct"
// MYSTRUCT-NEXT:   },
// MYSTRUCT-NEXT:   {
// MYSTRUCT-NEXT:     "kind": "text",
// MYSTRUCT-NEXT:     "spelling": ";"
// MYSTRUCT-NEXT:   }
// MYSTRUCT-NEXT: ]
// MYSTRUCT:      "kind": {
// MYSTRUCT-NEXT:   "displayName": "Structure",
// MYSTRUCT-NEXT:   "identifier": "c.struct"
// MYSTRUCT: "title": "MyStruct"
// MYSTRUCT:      "pathComponents": [
// MYSTRUCT-NEXT:    "MyStruct"
// MYSTRUCT-NEXT:  ]

// RUN: Filecheck %s --input-file %t/typedefchain.symbols.json --check-prefix MYSTRUCTSTRUCT
typedef MyStruct MyStructStruct;
// MYSTRUCTSTRUCT-LABEL: "!testLabel": "c:typedef_anonymous_record.c@T@MyStructStruct"
// MYSTRUCTSTRUCT: "accessLevel": "public",
// MYSTRUCTSTRUCT:     "declarationFragments": [
// MYSTRUCTSTRUCT-NEXT:  {
// MYSTRUCTSTRUCT-NEXT:    "kind": "keyword",
// MYSTRUCTSTRUCT-NEXT:    "spelling": "typedef"
// MYSTRUCTSTRUCT-NEXT:  },
// MYSTRUCTSTRUCT-NEXT:  {
// MYSTRUCTSTRUCT-NEXT:    "kind": "text",
// MYSTRUCTSTRUCT-NEXT:    "spelling": " "
// MYSTRUCTSTRUCT-NEXT:  },
// MYSTRUCTSTRUCT-NEXT:  {
// MYSTRUCTSTRUCT-NEXT:    "kind": "typeIdentifier",
// MYSTRUCTSTRUCT-NEXT:    "preciseIdentifier": "c:@SA@MyStruct",
// MYSTRUCTSTRUCT-NEXT:    "spelling": "MyStruct"
// MYSTRUCTSTRUCT-NEXT:  },
// MYSTRUCTSTRUCT-NEXT:  {
// MYSTRUCTSTRUCT-NEXT:    "kind": "text",
// MYSTRUCTSTRUCT-NEXT:    "spelling": " "
// MYSTRUCTSTRUCT-NEXT:  },
// MYSTRUCTSTRUCT-NEXT:  {
// MYSTRUCTSTRUCT-NEXT:    "kind": "identifier",
// MYSTRUCTSTRUCT-NEXT:    "spelling": "MyStructStruct"
// MYSTRUCTSTRUCT-NEXT:  },
// MYSTRUCTSTRUCT-NEXT:  {
// MYSTRUCTSTRUCT-NEXT:    "kind": "text",
// MYSTRUCTSTRUCT-NEXT:    "spelling": ";"
// MYSTRUCTSTRUCT-NEXT:  }
// MYSTRUCTSTRUCT-NEXT:],
// MYSTRUCTSTRUCT:     "kind": {
// MYSTRUCTSTRUCT-NEXT:  "displayName": "Type Alias",
// MYSTRUCTSTRUCT-NEXT:  "identifier": "c.typealias"

// RUN: Filecheck %s --input-file %t/typedefchain.symbols.json --check-prefix MYENUM
// RUN: Filecheck %s --input-file %t/typedefchain.symbols.json --check-prefix CASE
typedef enum { Case } MyEnum;
// MYENUM: "source": "c:@EA@MyEnum@Case",
// MYENUM-NEXT: "target": "c:@EA@MyEnum",
// MYENUM-NEXT: "targetFallback": "MyEnum"
// MYENUM-LABEL: "!testLabel": "c:@EA@MyEnum"
// MYENUM:     "declarationFragments": [
// MYENUM-NEXT:  {
// MYENUM-NEXT:    "kind": "keyword",
// MYENUM-NEXT:    "spelling": "typedef"
// MYENUM-NEXT:  },
// MYENUM-NEXT:  {
// MYENUM-NEXT:    "kind": "text",
// MYENUM-NEXT:    "spelling": " "
// MYENUM-NEXT:  },
// MYENUM-NEXT:  {
// MYENUM-NEXT:    "kind": "keyword",
// MYENUM-NEXT:    "spelling": "enum"
// MYENUM-NEXT:  },
// MYENUM-NEXT:  {
// MYENUM-NEXT:    "kind": "text",
// MYENUM-NEXT:    "spelling": " "
// MYENUM-NEXT:  },
// MYENUM-NEXT:  {
// MYENUM-NEXT:    "kind": "identifier",
// MYENUM-NEXT:    "spelling": "MyEnum"
// MYENUM-NEXT:  },
// MYENUM-NEXT:  {
// MYENUM-NEXT:    "kind": "text",
// MYENUM-NEXT:    "spelling": ";"
// MYENUM-NEXT:  }
// MYENUM-NEXT:],
// MYENUM:     "kind": {
// MYENUM-NEXT:  "displayName": "Enumeration",
// MYENUM-NEXT:  "identifier": "c.enum"
// MYENUM: "title": "MyEnum"

// CASE-LABEL: "!testLabel": "c:@EA@MyEnum@Case"
// CASE:      "pathComponents": [
// CASE-NEXT:   "MyEnum",
// CASE-NEXT:   "Case"
// CASE-NEXT: ]

// RUN: Filecheck %s --input-file %t/typedefchain.symbols.json --check-prefix MYENUMENUM
typedef MyEnum MyEnumEnum;
// MYENUMENUM-LABEL: "!testLabel": "c:typedef_anonymous_record.c@T@MyEnumEnum"
// MYENUMENUM:      "declarationFragments": [
// MYENUMENUM-NEXT:   {
// MYENUMENUM-NEXT:     "kind": "keyword",
// MYENUMENUM-NEXT:     "spelling": "typedef"
// MYENUMENUM-NEXT:   },
// MYENUMENUM-NEXT:   {
// MYENUMENUM-NEXT:     "kind": "text",
// MYENUMENUM-NEXT:     "spelling": " "
// MYENUMENUM-NEXT:   },
// MYENUMENUM-NEXT:   {
// MYENUMENUM-NEXT:     "kind": "typeIdentifier",
// MYENUMENUM-NEXT:     "preciseIdentifier": "c:@EA@MyEnum",
// MYENUMENUM-NEXT:     "spelling": "MyEnum"
// MYENUMENUM-NEXT:   },
// MYENUMENUM-NEXT:   {
// MYENUMENUM-NEXT:     "kind": "text",
// MYENUMENUM-NEXT:     "spelling": " "
// MYENUMENUM-NEXT:   },
// MYENUMENUM-NEXT:   {
// MYENUMENUM-NEXT:     "kind": "identifier",
// MYENUMENUM-NEXT:     "spelling": "MyEnumEnum"
// MYENUMENUM-NEXT:   },
// MYENUMENUM-NEXT:   {
// MYENUMENUM-NEXT:     "kind": "text",
// MYENUMENUM-NEXT:     "spelling": ";"
// MYENUMENUM-NEXT:   }
// MYENUMENUM-NEXT: ],
// MYENUMENUM:      "kind": {
// MYENUMENUM-NEXT:   "displayName": "Type Alias",
// MYENUMENUM-NEXT:   "identifier": "c.typealias"
// MYENUMENUM-NEXT: },
// MYENUMENUM: "title": "MyEnumEnum"

// expected-no-diagnostics
