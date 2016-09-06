// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@interface A
+ (id)alloc;
+ (id)init;
+ (id)new;
+ (Class)class;
+ (Class)superclass;
- (id)retain;
- (id)autorelease;
- (id)superclass;
@end

@interface B : A
- (int)B_method;
@end

@interface Unrelated
+ (id)icky;
@end

void message_id(B *b) {
  [[A alloc] init];
  [[b retain] B_method];
  [[b superclass] B_method];
}

@implementation Unrelated
+ (id)alloc {
  return [A alloc];
}
@end

@protocol P1
- (int)P1_method1;
+ (int)P1_method2;
@end

@protocol P2 <P1>
- (int)P2_method1;
+ (int)P2_method2;
@end

void message_qualified_id(id<P2> ip2) {
  [ip2 P1_method];
   ip2 P1_method];
}

// RUN: c-index-test -code-completion-at=%s:24:14 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCInstanceMethodDecl:{ResultType id}{TypedText autorelease}
// CHECK-CC1-NOT: B_method
// CHECK-CC1: ObjCInstanceMethodDecl:{ResultType id}{TypedText retain}
// RUN: c-index-test -code-completion-at=%s:25:15 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType id}{TypedText autorelease}
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType int}{TypedText B_method}
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType id}{TypedText retain}
// RUN: c-index-test -code-completion-at=%s:26:19 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType id}{TypedText autorelease}
// CHECK-CC3-NOT: B_method
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType id}{TypedText retain}


// RUN: c-index-test -code-completion-at=%s:31:13 %s | FileCheck -check-prefix=CHECK-SELECTOR-PREF %s
// CHECK-SELECTOR-PREF: ObjCClassMethodDecl:{ResultType id}{TypedText alloc} (32)
// CHECK-SELECTOR-PREF: ObjCClassMethodDecl:{ResultType Class}{TypedText class} (35)
// CHECK-SELECTOR-PREF: ObjCClassMethodDecl:{ResultType id}{TypedText init} (35)
// CHECK-SELECTOR-PREF: ObjCClassMethodDecl:{ResultType id}{TypedText new} (35)
// CHECK-SELECTOR-PREF: ObjCClassMethodDecl:{ResultType Class}{TypedText superclass} (35)

// RUN: c-index-test -code-completion-at=%s:46:8 %s | FileCheck -check-prefix=CHECK-INSTANCE-QUAL-ID %s
// RUN: c-index-test -code-completion-at=%s:47:8 %s | FileCheck -check-prefix=CHECK-INSTANCE-QUAL-ID %s
// CHECK-INSTANCE-QUAL-ID: ObjCInstanceMethodDecl:{ResultType int}{TypedText P1_method1} (37)
// CHECK-INSTANCE-QUAL-ID: ObjCInstanceMethodDecl:{ResultType int}{TypedText P2_method1} (35)
