// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %s -o - -verify | Filecheck %s

@protocol Protocol
@property(class) int myProtocolTypeProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(pl)Protocol(cpy)myProtocolTypeProp $ c:objc(pl)Protocol"
@property int myProtocolInstanceProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(pl)Protocol(py)myProtocolInstanceProp $ c:objc(pl)Protocol"
@end

@interface Interface
@property(class) int myInterfaceTypeProp;
// CHECk-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(cpy)myInterfaceTypeProp $ c:objc(cs)Interface"
@property int myInterfaceInstanceProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(py)myInterfaceInstanceProp $ c:objc(cs)Interface"
@end

@interface Interface (Category) <Protocol>
@property(class) int myCategoryTypeProp;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(cpy)myCategoryTypeProp $ c:objc(cs)Interface"
@property int myCategoryInstanceProp;
// CHECK-DAG "!testRelLabel": "memberOf $ c:objc(cs)Interface(py)myCategoryInstanceProp $ c:objc(cs)Interface"
@end

// expected-no-diagnostics
