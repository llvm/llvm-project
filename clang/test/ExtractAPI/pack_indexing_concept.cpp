// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++2d -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:  -triple arm64-apple-macosx -x c++-header %s -o %t/pack-indexing.symbols.json -verify

// RUN: FileCheck %s --input-file %t/pack-indexing.symbols.json --check-prefix FUNCTION

template <template <class> concept... CC, CC...[0] T>
void function(T);

// expected-no-diagnostics

//FUNCTION-LABEL: "!testLabel": "c:@FT@>2#pt>1#T#Tfunction#t0.1#v#"
//FUNCTION:      "declarationFragments": [
//FUNCTION:        "kind": "genericParameter",
//FUNCTION-NEXT:   "spelling": "CC"
//FUNCTION-NEXT: },
//FUNCTION-NEXT: {
//FUNCTION-NEXT:     "kind": "text",
//FUNCTION-NEXT:     "spelling": ", "
//FUNCTION-NEXT:   },
//FUNCTION-NEXT:   {
//FUNCTION-NEXT:     "kind": "typeIdentifier",
//FUNCTION-NEXT:     "spelling": "CC...[0]"
//FUNCTION-NEXT:   },
//FUNCTION-NEXT:   {
//FUNCTION-NEXT:     "kind": "text",
//FUNCTION-NEXT:     "spelling": " "
//FUNCTION-NEXT:   },
//FUNCTION-NEXT:   {
//FUNCTION-NEXT:     "kind": "genericParameter",
//FUNCTION-NEXT:     "spelling": "T"
//FUNCTION-NEXT:   },
