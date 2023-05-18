@interface Other {
@private
  int other_private;
@protected
  int other_protected;
@public
  int other_public;
}
@end

@interface Super {
@private
  int super_private;
@protected
  int super_protected;
@public
  int super_public;
}
@end

@interface Super () {
@private
  int super_ext_private;
@protected
  int super_ext_protected;
@public
  int super_ext_public;
}
@end

@interface Sub : Super {
@private
  int sub_private;
@protected
  int sub_protected;
@public
  int sub_public;
}
@end

@implementation Sub
- (void)method:(Sub *)sub with:(Other *)other {
  sub->super_protected = 1;
  other->other_public = 1;
}

void f(Sub *sub, Other *other) {
  sub->super_protected = 1;
  other->other_public = 1;
}
@end

// RUN: c-index-test -code-completion-at=%s:43:8 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-SUB %s
// RUN: c-index-test -code-completion-at=%s:48:8 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-SUB %s
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText sub_private} (35)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText sub_protected} (35)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText sub_public} (35)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText super_ext_private} (35) (inaccessible)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText super_ext_protected} (35)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText super_ext_public} (35)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText super_private} (37) (inaccessible)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText super_protected} (37)
// CHECK-SUB: ObjCIvarDecl:{ResultType int}{TypedText super_public} (37)

// RUN: c-index-test -code-completion-at=%s:44:10 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-OTHER %s
// RUN: c-index-test -code-completion-at=%s:49:10 -fobjc-nonfragile-abi %s | FileCheck -check-prefix=CHECK-OTHER %s
// CHECK-OTHER: ObjCIvarDecl:{ResultType int}{TypedText other_private} (35) (inaccessible)
// CHECK-OTHER: ObjCIvarDecl:{ResultType int}{TypedText other_protected} (35) (inaccessible)
// CHECK-OTHER: ObjCIvarDecl:{ResultType int}{TypedText other_public} (35)
