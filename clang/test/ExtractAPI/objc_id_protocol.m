// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -x objective-c-header -triple arm64-apple-macosx %s -o - -verify | FileCheck %s

@protocol MyProtocol
@end

@interface MyInterface
@property(copy, readwrite) id<MyProtocol> obj1;
// CHECK-LABEL: "!testLabel": "c:objc(cs)MyInterface(py)obj1"
// CHECK:      "declarationFragments": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "keyword",
// CHECK-NEXT:     "spelling": "@property"
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "text",
// CHECK-NEXT:     "spelling": " ("
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "keyword",
// CHECK-NEXT:     "spelling": "copy"
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "text",
// CHECK-NEXT:     "spelling": ", "
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "keyword",
// CHECK-NEXT:     "spelling": "readwrite"
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "text",
// CHECK-NEXT:     "spelling": ") "
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "typeIdentifier",
// CHECK-NEXT:     "preciseIdentifier": "c:Qoobjc(pl)MyProtocol",
// CHECK-NEXT:     "spelling": "id<MyProtocol>"
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "text",
// CHECK-NEXT:     "spelling": " "
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "identifier",
// CHECK-NEXT:     "spelling": "obj1"
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:     "kind": "text",
// CHECK-NEXT:     "spelling": ";"
// CHECK-NEXT:   }
// CHECK-NEXT: ],
@end

// expected-no-diagnostics
