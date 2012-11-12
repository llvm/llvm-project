/* Note: the RUN lines are near the end of the file, since line/column
 matter for this test. */
@class MyClass;
@interface I1 
{
  id StoredProp3;
  int RandomIVar;
}
@property int Prop0;
@property int Prop1;
@property float Prop2;
@end

@interface I2 : I1
@property id Prop3;
@property id Prop4;
@end

@implementation I2
@synthesize Prop2, Prop1, Prop3 = StoredProp3;
@dynamic Prop4;
@end

@interface I3 : I2
@property id Prop3;
@end

id test(I3 *i3) {
  return i3.Prop3;
}

@interface I4
@property id Prop2;
@end

@interface I4 () {
  I4 *Prop1;
}
@end

@implementation I4 {
  id Prop2_;
}

@synthesize Prop2 = Prop2_;
@end

@protocol P1
@property int Prop5;
@end

@class P1;

@interface I5<P1>
@end
@implementation I5
@synthesize Prop5;
@end
// RUN: c-index-test -code-completion-at=%s:20:13 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText Prop0}
// CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText Prop1}
// CHECK-CC1: ObjCPropertyDecl:{ResultType float}{TypedText Prop2}
// CHECK-CC1: ObjCPropertyDecl:{ResultType id}{TypedText Prop3}
// CHECK-CC1: ObjCPropertyDecl:{ResultType id}{TypedText Prop4}
// RUN: c-index-test -code-completion-at=%s:20:20 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCPropertyDecl:{ResultType int}{TypedText Prop0}
// CHECK-CC2: ObjCPropertyDecl:{ResultType int}{TypedText Prop1}
// CHECK-CC2-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop3}
// CHECK-CC2: ObjCPropertyDecl:{ResultType id}{TypedText Prop4}
// RUN: c-index-test -code-completion-at=%s:20:35 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCIvarDecl:{ResultType id}{TypedText _Prop3} (36)
// CHECK-CC3: ObjCIvarDecl:{ResultType int}{TypedText RandomIVar} (35)
// CHECK-CC3: ObjCIvarDecl:{ResultType id}{TypedText StoredProp3} (8)

// RUN: c-index-test -code-completion-at=%s:21:10 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCPropertyDecl:{ResultType int}{TypedText Prop0}
// CHECK-CC4-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop4}

// RUN: c-index-test -code-completion-at=%s:29:13 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCPropertyDecl:{ResultType int}{TypedText Prop0} (35)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText Prop1} (35)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType float}{TypedText Prop2} (35)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop3} (35)
// CHECK-CC5-NEXT: ObjCPropertyDecl:{ResultType id}{TypedText Prop4} (35)

// RUN: c-index-test -code-completion-at=%s:9:11 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInterfaceDecl:{TypedText MyClass} (50)


// RUN: c-index-test -code-completion-at=%s:45:21 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7-NOT: ObjCIvarDecl:{ResultType id}{TypedText _Prop2}
// CHECK-CC7: ObjCIvarDecl:{ResultType I4 *}{TypedText Prop1} (17)
// CHECK-CC7: ObjCIvarDecl:{ResultType id}{TypedText Prop2_} (7)

// RUN: c-index-test -code-completion-at=%s:57:13 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: ObjCPropertyDecl:{ResultType int}{TypedText Prop5} (35)
