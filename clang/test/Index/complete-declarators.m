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
- (boid)method2 {}
@end

// RUN: c-index-test -code-completion-at=%s:7:4 %s | FileCheck -check-prefix=CHECK-CC0 %s
// CHECK-CC0: Pattern:{TypedText IBAction}{RightParen )}{Placeholder selector}{Colon :}{LeftParen (}{Text id}{RightParen )}{Text sender} (40)
// CHECK-CC0: macro definition:{TypedText IBAction} (70)
// CHECK-CC0: macro definition:{TypedText IBOutlet} (70)
// CHECK-CC0: macro definition:{TypedText IBOutletCollection}{LeftParen (}{Placeholder ClassName}{RightParen )} (70)
// CHECK-CC0: TypedefDecl:{TypedText id} (50)
// CHECK-CC0: Keyword:{TypedText in} (40)
// CHECK-CC0: Keyword:{TypedText inout} (40)
// CHECK-CC0: Keyword:{TypedText instancetype} (40)
// CHECK-CC0: Keyword:{TypedText int} (50)
// CHECK-CC0: Keyword:{TypedText long} (50)
// RUN: c-index-test -code-completion-at=%s:7:19 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1-NOT: Keyword:{TypedText extern} (40)
// CHECK-CC1: Pattern:{TypedText param1} (40)
// RUN: c-index-test -code-completion-at=%s:9:15 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:15:10 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:16:9 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: Keyword:{TypedText const} (40)
// CHECK-CC2-NOT: int
// CHECK-CC2: Keyword:{TypedText restrict} (40)
// CHECK-CC2: Keyword:{TypedText volatile} (40)
// RUN: c-index-test -code-completion-at=%s:15:15 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ParmDecl:{ResultType id}{TypedText param1} (34)
// CHECK-CC3-NOT: VarDecl:{ResultType int}{TypedText q2}
// CHECK-CC3-NOT: VarDecl:{ResultType id}{TypedText q}
// CHECK-CC3: Declaration:{ResultType A *}{TypedText self} (34)
// CHECK-CC3: Pattern:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: c-index-test -code-completion-at=%s:15:15 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ParmDecl:{ResultType id}{TypedText param1} (34)
// CHECK-CC4-NOT: VarDecl:{ResultType int}{TypedText q2}
// CHECK-CC4: Declaration:{ResultType A *}{TypedText self} (34)
// CHECK-CC4: Pattern:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: c-index-test -code-completion-at=%s:23:10 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: Keyword:{TypedText _Bool} (50)
// CHECK-CC5: Keyword:{TypedText _Complex} (50)
// CHECK-CC5: Keyword:{TypedText _Imaginary} (50)
// CHECK-CC5: ObjCInterfaceDecl:{TypedText A} (50)
// CHECK-CC5: Keyword:{TypedText char} (50)
// CHECK-CC5: TypedefDecl:{TypedText Class} (50)
// CHECK-CC5: Keyword:{TypedText const} (50)
// CHECK-CC5: Keyword:{TypedText double} (50)
// CHECK-CC5: Keyword:{TypedText enum} (50)
// CHECK-CC5: Keyword:{TypedText float} (50)
// CHECK-CC5: TypedefDecl:{TypedText id} (50)
// CHECK-CC5: Keyword:{TypedText int} (50)
// CHECK-CC5: Keyword:{TypedText long} (50)
// CHECK-CC5: Keyword:{TypedText restrict} (50)
// CHECK-CC5: TypedefDecl:{TypedText SEL} (50)
// CHECK-CC5: Keyword:{TypedText short} (50)
// CHECK-CC5: Keyword:{TypedText signed} (50)
// CHECK-CC5: Keyword:{TypedText struct} (50)
// CHECK-CC5: Pattern:{TypedText typeof}{HorizontalSpace  }{Placeholder expression} (40)
// CHECK-CC5: Pattern:{TypedText typeof}{LeftParen (}{Placeholder type}{RightParen )} (40)
// CHECK-CC5: Keyword:{TypedText union} (50)
// CHECK-CC5: Keyword:{TypedText unsigned} (50)
// CHECK-CC5: Keyword:{TypedText void} (50)
// CHECK-CC5: Keyword:{TypedText volatile} (50)

// Check that there are no duplicate entries if we code-complete after an @implementation
// RUN: c-index-test -code-completion-at=%s:27:1 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInterfaceDecl:{TypedText A}
// CHECK-CC6-NOT: ObjCInterfaceDecl:{TypedText A}
