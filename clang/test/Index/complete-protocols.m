/* Note: the RUN lines are near the end of the file, since line/column
   matter for this test. */

@protocol Protocol1
@end

@protocol Protocol2;

void f(id<Protocol1,Protocol2>);

@protocol Protocol0;
@protocol NewProtocol 
{
}
@end

// RUN: c-index-test -code-completion-at=%s:9:11 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCProtocolDecl:{TypedText Protocol1}
// CHECK-CC1-NEXT: ObjCProtocolDecl:{TypedText Protocol2}
// RUN: c-index-test -code-completion-at=%s:9:21 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2-NOT: ObjCProtocolDecl:{TypedText Protocol1}
// CHECK-CC2: ObjCProtocolDecl:{TypedText Protocol2}
// RUN: c-index-test -code-completion-at=%s:12:11 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCProtocolDecl:{TypedText Protocol0}
// CHECK-CC3-NEXT: ObjCProtocolDecl:{TypedText Protocol2}

// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:9:11 %s | FileCheck -check-prefix=CHECK-CC1 %s
