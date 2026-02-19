// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %s -o %t/output.symbols.json -verify
// RUN: FileCheck %s --input-file %t/output.symbols.json

@protocol Protocol
@property(class) int myProtocolTypeProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(pl)Protocol(cpy)myProtocolTypeProp $ c:objc(pl)Protocol"
@property int myProtocolInstanceProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(pl)Protocol(py)myProtocolInstanceProp $ c:objc(pl)Protocol"
@end

@interface Interface
@property(class) int myInterfaceTypeProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(cpy)myInterfaceTypeProp $ c:objc(cs)Interface"
@property int myInterfaceInstanceProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(py)myInterfaceInstanceProp $ c:objc(cs)Interface"

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix NULLABLE
@property(nullable, strong) id myNullableProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(py)myNullableProp $ c:objc(cs)Interface"
// NULLABLE: "!testLabel": "c:objc(cs)Interface(py)myNullableProp"
// NULLABLE:      "declarationFragments": [
// NULLABLE:          "kind": "keyword",
// NULLABLE-NEXT:     "spelling": "@property"
// NULLABLE:          "kind": "keyword",
// NULLABLE-NEXT:     "spelling": "strong"
// NULLABLE:          "kind": "keyword",
// NULLABLE-NEXT:     "spelling": "nullable"

// RUN: FileCheck %s --input-file %t/output.symbols.json --check-prefix NULLRESETTABLE
@property(null_resettable, strong) id myNullResettableProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(py)myNullResettableProp $ c:objc(cs)Interface"
// NULLRESETTABLE: "!testLabel": "c:objc(cs)Interface(py)myNullResettableProp"
// NULLABLE:      "declarationFragments": [
// NULLABLE:          "kind": "keyword",
// NULLABLE-NEXT:     "spelling": "@property"
// NULLABLE:          "kind": "keyword",
// NULLABLE-NEXT:     "spelling": "strong"
// NULLABLE:          "kind": "keyword",
// NULLABLE-NEXT:     "spelling": "null_resettable"
@end

@interface Interface (Category) <Protocol>
@property(class) int myCategoryTypeProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(cpy)myCategoryTypeProp $ c:objc(cs)Interface"
@property int myCategoryInstanceProp;
// CHECK-DAG "!testRelLabel": "memberOf $ c:objc(cs)Interface(py)myCategoryInstanceProp $ c:objc(cs)Interface"
@end

// expected-no-diagnostics
