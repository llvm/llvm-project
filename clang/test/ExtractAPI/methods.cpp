// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -x c++-header %s -o %t/output.symbols.json -verify

class Foo {
  // RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix GETCOUNT
  int getCount();
  // GETCOUNT: "!testRelLabel": "memberOf $ c:@S@Foo@F@getCount# $ c:@S@Foo"
  // GETCOUNT-LABEL: "!testLabel":  "c:@S@Foo@F@getCount#"
  // GETCOUNT: "accessLevel": "private",
  // GETCOUNT:      "declarationFragments": [
  // GETCOUNT-NEXT:   {
  // GETCOUNT-NEXT:     "kind": "typeIdentifier",
  // GETCOUNT-NEXT:     "preciseIdentifier": "c:I",
  // GETCOUNT-NEXT:     "spelling": "int"
  // GETCOUNT-NEXT:   },
  // GETCOUNT-NEXT:   {
  // GETCOUNT-NEXT:     "kind": "text",
  // GETCOUNT-NEXT:     "spelling": " "
  // GETCOUNT-NEXT:   },
  // GETCOUNT-NEXT:   {
  // GETCOUNT-NEXT:     "kind": "identifier",
  // GETCOUNT-NEXT:     "spelling": "getCount"
  // GETCOUNT-NEXT:   },
  // GETCOUNT-NEXT:   {
  // GETCOUNT-NEXT:     "kind": "text",
  // GETCOUNT-NEXT:     "spelling": "();"
  // GETCOUNT-NEXT:   }
  // GETCOUNT-NEXT: ],
  // GETCOUNT:      "functionSignature": {
  // GETCOUNT-NEXT:   "returns": [
  // GETCOUNT-NEXT:     {
  // GETCOUNT-NEXT:       "kind": "typeIdentifier",
  // GETCOUNT-NEXT:       "preciseIdentifier": "c:I",
  // GETCOUNT-NEXT:       "spelling": "int"
  // GETCOUNT-NEXT:     }
  // GETCOUNT-NEXT:   ]
  // GETCOUNT-NEXT: },
  // GETCOUNT: "displayName": "Instance Method",
  // GETCOUNT-NEXT: "identifier": "c++.method"
  // GETCOUNT: "title": "getCount"
  // GETCOUNT: "pathComponents": [
  // GETCOUNT-NEXT:   "Foo",
  // GETCOUNT-NEXT:   "getCount"
  // GETCOUNT-NEXT: ]

  // RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix SETL
  void setLength(int length) noexcept;
  // SETL: "!testRelLabel": "memberOf $ c:@S@Foo@F@setLength#I# $ c:@S@Foo"
  // SETL-LABEL: "!testLabel": "c:@S@Foo@F@setLength#I#"
  // SETL:      "declarationFragments": [
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "typeIdentifier",
  // SETL-NEXT:     "preciseIdentifier": "c:v",
  // SETL-NEXT:     "spelling": "void"
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "text",
  // SETL-NEXT:     "spelling": " "
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "identifier",
  // SETL-NEXT:     "spelling": "setLength"
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "text",
  // SETL-NEXT:     "spelling": "("
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "typeIdentifier",
  // SETL-NEXT:     "preciseIdentifier": "c:I",
  // SETL-NEXT:     "spelling": "int"
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "text",
  // SETL-NEXT:     "spelling": " "
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "internalParam",
  // SETL-NEXT:     "spelling": "length"
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "text",
  // SETL-NEXT:     "spelling": ")"
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "text",
  // SETL-NEXT:     "spelling": " "
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "keyword",
  // SETL-NEXT:     "spelling": "noexcept"
  // SETL-NEXT:   },
  // SETL-NEXT:   {
  // SETL-NEXT:     "kind": "text",
  // SETL-NEXT:     "spelling": ";"
  // SETL-NEXT:   }
  // SETL-NEXT: ],
  // SETL:      "functionSignature": {
  // SETL-NEXT:   "parameters": [
  // SETL-NEXT:     {
  // SETL-NEXT:       "declarationFragments": [
  // SETL-NEXT:         {
  // SETL-NEXT:           "kind": "typeIdentifier",
  // SETL-NEXT:           "preciseIdentifier": "c:I",
  // SETL-NEXT:           "spelling": "int"
  // SETL-NEXT:         },
  // SETL-NEXT:         {
  // SETL-NEXT:           "kind": "text",
  // SETL-NEXT:           "spelling": " "
  // SETL-NEXT:         },
  // SETL-NEXT:         {
  // SETL-NEXT:           "kind": "internalParam",
  // SETL-NEXT:           "spelling": "length"
  // SETL-NEXT:         }
  // SETL-NEXT:       ],
  // SETL-NEXT:       "name": "length"
  // SETL-NEXT:     }
  // SETL-NEXT:   ],
  // SETL-NEXT:   "returns": [
  // SETL-NEXT:     {
  // SETL-NEXT:       "kind": "typeIdentifier",
  // SETL-NEXT:       "preciseIdentifier": "c:v",
  // SETL-NEXT:       "spelling": "void"
  // SETL-NEXT:     }
  // SETL-NEXT:   ]
  // SETL-NEXT: },

public:
  // RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix GETFOO
  static double getFoo();
  // GETFOO: "!testRelLabel": "memberOf $ c:@S@Foo@F@getFoo#S $ c:@S@Foo"

  // GETFOO-LABEL: "!testLabel": "c:@S@Foo@F@getFoo#S"
  // GETFOO: "accessLevel": "public",
  // GETFOO:      "declarationFragments": [
  // GETFOO-NEXT:   {
  // GETFOO-NEXT:     "kind": "keyword",
  // GETFOO-NEXT:     "spelling": "static"
  // GETFOO-NEXT:   },
  // GETFOO-NEXT:   {
  // GETFOO-NEXT:     "kind": "text",
  // GETFOO-NEXT:     "spelling": " "
  // GETFOO-NEXT:   },
  // GETFOO-NEXT:   {
  // GETFOO-NEXT:     "kind": "typeIdentifier",
  // GETFOO-NEXT:     "preciseIdentifier": "c:d",
  // GETFOO-NEXT:     "spelling": "double"
  // GETFOO-NEXT:   },
  // GETFOO-NEXT:   {
  // GETFOO-NEXT:     "kind": "text",
  // GETFOO-NEXT:     "spelling": " "
  // GETFOO-NEXT:   },
  // GETFOO-NEXT:   {
  // GETFOO-NEXT:     "kind": "identifier",
  // GETFOO-NEXT:     "spelling": "getFoo"
  // GETFOO-NEXT:   },
  // GETFOO-NEXT:   {
  // GETFOO-NEXT:     "kind": "text",
  // GETFOO-NEXT:     "spelling": "();"
  // GETFOO-NEXT:   }
  // GETFOO-NEXT: ],
  // GETFOO:      "functionSignature": {
  // GETFOO-NEXT:   "returns": [
  // GETFOO-NEXT:     {
  // GETFOO-NEXT:       "kind": "typeIdentifier",
  // GETFOO-NEXT:       "preciseIdentifier": "c:d",
  // GETFOO-NEXT:       "spelling": "double"
  // GETFOO-NEXT:     }
  // GETFOO-NEXT:   ]
  // GETFOO-NEXT: },
  // GETFOO:      "kind": {
  // GETFOO-NEXT:   "displayName": "Static Method",
  // GETFOO-NEXT:   "identifier": "c++.type.method"
  // GETFOO-NEXT: },

protected:
  // RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix GETBAR
  constexpr int getBar() const;
  // GETBAR: "!testRelLabel": "memberOf $ c:@S@Foo@F@getBar#1 $ c:@S@Foo"

  // GETBAR-LABEL: "!testLabel": "c:@S@Foo@F@getBar#1"
  // GETBAR: "accessLevel": "protected"
  // GETBAR:      "declarationFragments": [
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "keyword",
  // GETBAR-NEXT:     "spelling": "constexpr"
  // GETBAR-NEXT:   },
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "text",
  // GETBAR-NEXT:     "spelling": " "
  // GETBAR-NEXT:   },
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "typeIdentifier",
  // GETBAR-NEXT:     "preciseIdentifier": "c:I",
  // GETBAR-NEXT:     "spelling": "int"
  // GETBAR-NEXT:   },
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "text",
  // GETBAR-NEXT:     "spelling": " "
  // GETBAR-NEXT:   },
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "identifier",
  // GETBAR-NEXT:     "spelling": "getBar"
  // GETBAR-NEXT:   },
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "text",
  // GETBAR-NEXT:     "spelling": "() "
  // GETBAR-NEXT:   },
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "keyword",
  // GETBAR-NEXT:     "spelling": "const"
  // GETBAR-NEXT:   },
  // GETBAR-NEXT:   {
  // GETBAR-NEXT:     "kind": "text",
  // GETBAR-NEXT:     "spelling": ";"
  // GETBAR-NEXT:   }
  // GETBAR-NEXT: ],
};

// expected-no-diagnostics
