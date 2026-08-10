// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@interface Super { }
- (int)getter1;
+ (int)getter2_not_instance;
- (int)getter2_not:(int)x;
- (int)getter3;
- (void)setter1:(int)x;
+ (void)setter2_not_inst:(int)x;
+ (void)setter2_many_args:(int)x second:(int)y;
- (void)setter3:(int)y;
@property (getter = getter1, setter = setter1:) int blah;
@end

@interface Sub : Super { }
- (int)getter4;
- (void)setter4:(int)x;
@property (getter = getter4, setter = setter1:) int blarg;
@end

// RUN: c-index-test -code-completion-at=%s:13:21 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCInstanceMethodDecl:{ResultType int}{TypedText getter1}
// CHECK-CC1-NOT: getter2
// CHECK-CC1: ObjCInstanceMethodDecl:{ResultType int}{TypedText getter3}
// RUN: c-index-test -code-completion-at=%s:13:39 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType int}{TypedText getter2_not:}
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType void}{TypedText setter1:}
// CHECK-CC2-NOT: setter2
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType void}{TypedText setter3:}
// RUN: c-index-test -code-completion-at=%s:19:21 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType int}{TypedText getter1}
// CHECK-CC3-NOT: getter2
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType int}{TypedText getter3}
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType int}{TypedText getter4}
// RUN: c-index-test -code-completion-at=%s:19:39 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType int}{TypedText getter2_not:}{Informative (int)x}
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType void}{TypedText setter1:}{Informative (int)x}
// CHECK-CC4-NOT: setter2
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType void}{TypedText setter3:}{Informative (int)y}
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType void}{TypedText setter4:}{Informative (int)x}
