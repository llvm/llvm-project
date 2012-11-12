// This test is line- and column-sensitive, so test commands are at the bottom.
@protocol P
- (int)method:(id)param1;
@end

@interface A <P>
- (int)method:(id)param1;

@property int prop1;
@end

@implementation A
- (int)method:(id)param1 {
  int q2;
  for(id q in param1) {
    int y;
  }
  id q;
  for(q in param1) {
    int y;
  }

  static P *p = 0;
}
@end

// RUN: c-index-test -code-completion-at=%s:7:19 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1-NOT: NotImplemented:{TypedText extern} (40)
// CHECK-CC1: NotImplemented:{TypedText param1} (40)
// RUN: c-index-test -code-completion-at=%s:9:15 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:15:10 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:16:9 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText const} (40)
// CHECK-CC2-NOT: int
// CHECK-CC2: NotImplemented:{TypedText restrict} (40)
// CHECK-CC2: NotImplemented:{TypedText volatile} (40)
// RUN: c-index-test -code-completion-at=%s:15:15 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ParmDecl:{ResultType id}{TypedText param1} (34)
// CHECK-CC3-NOT: VarDecl:{ResultType int}{TypedText q2}
// CHECK-CC3-NOT: VarDecl:{ResultType id}{TypedText q}
// CHECK-CC3: NotImplemented:{ResultType A *}{TypedText self} (34)
// CHECK-CC3: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: c-index-test -code-completion-at=%s:15:15 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ParmDecl:{ResultType id}{TypedText param1} (34)
// CHECK-CC4-NOT: VarDecl:{ResultType int}{TypedText q2}
// CHECK-CC4: NotImplemented:{ResultType A *}{TypedText self} (34)
// CHECK-CC4: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: c-index-test -code-completion-at=%s:23:10 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: NotImplemented:{TypedText _Bool} (50)
// CHECK-CC5: NotImplemented:{TypedText _Complex} (50)
// CHECK-CC5: NotImplemented:{TypedText _Imaginary} (50)
// CHECK-CC5: ObjCInterfaceDecl:{TypedText A} (50)
// CHECK-CC5: NotImplemented:{TypedText char} (50)
// CHECK-CC5: TypedefDecl:{TypedText Class} (50)
// CHECK-CC5: NotImplemented:{TypedText const} (50)
// CHECK-CC5: NotImplemented:{TypedText double} (50)
// CHECK-CC5: NotImplemented:{TypedText enum} (50)
// CHECK-CC5: NotImplemented:{TypedText float} (50)
// CHECK-CC5: TypedefDecl:{TypedText id} (50)
// CHECK-CC5: NotImplemented:{TypedText int} (50)
// CHECK-CC5: NotImplemented:{TypedText long} (50)
// CHECK-CC5: NotImplemented:{TypedText restrict} (50)
// CHECK-CC5: TypedefDecl:{TypedText SEL} (50)
// CHECK-CC5: NotImplemented:{TypedText short} (50)
// CHECK-CC5: NotImplemented:{TypedText signed} (50)
// CHECK-CC5: NotImplemented:{TypedText struct} (50)
// CHECK-CC5: NotImplemented:{TypedText typeof}{HorizontalSpace  }{Placeholder expression} (40)
// CHECK-CC5: NotImplemented:{TypedText typeof}{LeftParen (}{Placeholder type}{RightParen )} (40)
// CHECK-CC5: NotImplemented:{TypedText union} (50)
// CHECK-CC5: NotImplemented:{TypedText unsigned} (50)
// CHECK-CC5: NotImplemented:{TypedText void} (50)
// CHECK-CC5: NotImplemented:{TypedText volatile} (50)
