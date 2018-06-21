// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 %s > %t/out
// RUN: FileCheck %s < %t/out

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

@protocol NSObject
@end

@interface NSObject
@end

// CHECK: <Declaration>@interface A &lt;__covariant T : id, U : NSObject *&gt; : NSObject
/// A
@interface A<__covariant T : id, U : NSObject *> : NSObject
@end

// CHECK: <Declaration>@interface AA : A &lt;id, NSObject *&gt;
/// AA
@interface AA : A<id, NSObject *>
@end
