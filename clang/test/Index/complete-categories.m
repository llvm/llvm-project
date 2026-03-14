/* Note: the RUN lines are near the end of the file, since line/column
   matter for this test. */

@interface I1 @end
@interface I2 @end
@interface I3 : I2 @end

@interface I1(Cat1) @end
@interface I1(Cat2) @end
@interface I1(Cat3) @end

@interface I2 (Cat2) @end
@interface I2 (Cat3) @end
@interface I2 (Cat2) @end
@interface I3 (Cat1) @end
@interface I3 (Cat2) @end

@implementation I1(Cat2) @end
@implementation I1(Cat3) @end
@implementation I3(Cat2) @end

// RUN: c-index-test -code-completion-at=%s:12:16 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCCategoryDecl:{TypedText Cat1}
// CHECK-CC1: ObjCCategoryDecl:{TypedText Cat2}
// CHECK-CC1: ObjCCategoryDecl:{TypedText Cat3}
// RUN: c-index-test -code-completion-at=%s:13:16 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCCategoryDecl:{TypedText Cat1}
// CHECK-CC2-NEXT: ObjCCategoryDecl:{TypedText Cat3}
// RUN: c-index-test -code-completion-at=%s:18:20 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCCategoryDecl:{TypedText Cat1}
// CHECK-CC3: ObjCCategoryDecl:{TypedText Cat2}
// CHECK-CC3: ObjCCategoryDecl:{TypedText Cat3}
// RUN: c-index-test -code-completion-at=%s:19:20 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCCategoryDecl:{TypedText Cat1}
// CHECK-CC4-NEXT: ObjCCategoryDecl:{TypedText Cat3}
// RUN: c-index-test -code-completion-at=%s:20:20 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCCategoryDecl:{TypedText Cat1}
// CHECK-CC5-NEXT: ObjCCategoryDecl:{TypedText Cat2}
// CHECK-CC5-NEXT: ObjCCategoryDecl:{TypedText Cat3}
