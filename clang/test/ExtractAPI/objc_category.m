// RUN: rm -rf %t
// RUN: %clang_cc1 -extract-api --pretty-sgf --emit-sgf-symbol-labels-for-testing \
// RUN:   -triple arm64-apple-macosx -x objective-c-header %s -o - -verify | FileCheck %s

@protocol Protocol
@end

@interface Interface
@end

@interface Interface (Category) <Protocol>
// CHECK-DAG: "!testRelLabel": "conformsTo $ c:objc(cs)Interface $ c:objc(pl)Protocol"
@property int Property;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(py)Property $ c:objc(cs)Interface"
- (void)InstanceMethod;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(im)InstanceMethod $ c:objc(cs)Interface"
+ (void)ClassMethod;
// CHECK-DAG: "!testRelLabel": "memberOf $ c:objc(cs)Interface(cm)ClassMethod $ c:objc(cs)Interface"
@end

// expected-no-diagnostics
