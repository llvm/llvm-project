/* Note: the RUN lines are near the end of the file, since line/column
   matter for this test. */

@class Int1, Int2, Int3, Int4;

@interface Int3 
{
}
@end

@interface Int2 : Int3
{
}
@end

@implementation Int2
@end

@implementation Int3
@end

// RUN: c-index-test -code-completion-at=%s:6:12 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCInterfaceDecl:{TypedText Int1}
// CHECK-CC1: ObjCInterfaceDecl:{TypedText Int2}
// CHECK-CC1: ObjCInterfaceDecl:{TypedText Int3}
// CHECK-CC1: ObjCInterfaceDecl:{TypedText Int4}
// RUN: c-index-test -code-completion-at=%s:11:12 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCInterfaceDecl:{TypedText Int1}
// CHECK-CC2-NEXT: ObjCInterfaceDecl:{TypedText Int2}
// CHECK-CC2-NEXT: ObjCInterfaceDecl:{TypedText Int3}
// CHECK-CC2-NEXT: ObjCInterfaceDecl:{TypedText Int4}
// RUN: c-index-test -code-completion-at=%s:11:19 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInterfaceDecl:{TypedText Int1}
// CHECK-CC3-NEXT: ObjCInterfaceDecl:{TypedText Int3}
// CHECK-CC3-NEXT: ObjCInterfaceDecl:{TypedText Int4}
// RUN: c-index-test -code-completion-at=%s:16:17 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInterfaceDecl:{TypedText Int1}
// CHECK-CC4-NEXT: ObjCInterfaceDecl:{TypedText Int2}
// CHECK-CC4-NEXT: ObjCInterfaceDecl:{TypedText Int3}
// CHECK-CC4-NEXT: ObjCInterfaceDecl:{TypedText Int4}
// RUN: c-index-test -code-completion-at=%s:19:17 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCInterfaceDecl:{TypedText Int1}
// CHECK-CC5-NEXT: ObjCInterfaceDecl:{TypedText Int3}
// CHECK-CC5-NEXT: ObjCInterfaceDecl:{TypedText Int4}


// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:11:12 %s | FileCheck -check-prefix=CHECK-CC2 %s


void useClasses() {
  int i = 0;
  [Int3 message:1];
}

// RUN: c-index-test -code-completion-at=%s:51:11 %s | FileCheck -check-prefix=CHECK-USE %s
// RUN: c-index-test -code-completion-at=%s:52:17 %s | FileCheck -check-prefix=CHECK-USE %s
// CHECK-USE: ObjCInterfaceDecl:{TypedText Int2} (50)
// CHECK-USE: ObjCInterfaceDecl:{TypedText Int3} (50)
// CHECK-USE-NOT: Int1
// CHECK-USE-NOT: Int4

// Caching should work too:
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:51:11 %s | FileCheck -check-prefix=CHECK-USE %s
