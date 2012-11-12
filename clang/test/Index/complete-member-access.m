/* Note: the RUN lines are near the end of the file, since line/column
   matter for this test. */

@protocol MyProtocol
@property float ProtoProp;
@end

@interface Super {
  int SuperIVar;
}
@end
@interface Int : Super<MyProtocol>
{
  int IVar;
}

@property int prop1;
@end

void test_props(Int* ptr) {
  ptr.prop1 = 0;
  ptr->IVar = 0;
}

@interface Sub : Int 
@property int myProp;

- (int)myProp;
- (int)myOtherPropLikeThing;
- (int)myOtherNonPropThing:(int)value;
@end

int test_more_props(Sub *s) {
  return s.myOtherPropLikeThing;
}

@interface Other
@property Sub *sub;
@end

int test_two_levels(Other *other) {
  return other.sub.myProp;
}

// RUN: c-index-test -code-completion-at=%s:21:7 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText prop1}
// CHECK-CC1: ObjCPropertyDecl:{ResultType float}{TypedText ProtoProp}
// CHECK-CC1: Completion contexts:
// CHECK-CC1-NEXT: Objective-C property access
// CHECK-CC1-NEXT: Container Kind: ObjCInterfaceDecl
// CHECK-CC1-NEXT: Container is complete
// CHECK-CC1-NEXT: Container USR: c:objc(cs)Int
// RUN: c-index-test -code-completion-at=%s:22:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCIvarDecl:{ResultType int}{TypedText IVar} (35)
// CHECK-CC2: ObjCIvarDecl:{ResultType int}{TypedText SuperIVar} (37)
// CHECK-CC2: Completion contexts:
// CHECK-CC2-NEXT: Arrow member access
// CHECK-CC2-NEXT: Container Kind: ObjCInterfaceDecl
// CHECK-CC2-NEXT: Container is complete
// CHECK-CC2-NEXT: Container USR: c:objc(cs)Int
// RUN: c-index-test -code-completion-at=%s:34:12 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType int}{TypedText myOtherPropLikeThing} (37)
// CHECK-CC3: ObjCPropertyDecl:{ResultType int}{TypedText myProp} (35)
// CHECK-CC3: ObjCPropertyDecl:{ResultType int}{TypedText prop1} (35)
// CHECK-CC3: ObjCPropertyDecl:{ResultType float}{TypedText ProtoProp} (35)
// CHECK-CC3: Completion contexts:
// CHECK-CC3-NEXT: Objective-C property access
// CHECK-CC3-NEXT: Container Kind: ObjCInterfaceDecl
// CHECK-CC3-NEXT: Container is complete
// CHECK-CC3-NEXT: Container USR: c:objc(cs)Sub

// RUN: c-index-test -code-completion-at=%s:42:20 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType int}{TypedText myOtherPropLikeThing} (37)
// CHECK-CC4-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText myProp} (35)
// CHECK-CC4-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText prop1} (35)
// CHECK-CC4-NEXT: ObjCPropertyDecl:{ResultType float}{TypedText ProtoProp} (35)

