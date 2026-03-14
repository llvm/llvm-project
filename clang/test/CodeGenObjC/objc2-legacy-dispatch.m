// RUN: %clang_cc1 -fobjc-dispatch-method=mixed -triple i386-apple-darwin10 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK_NEW_DISPATCH %s
//
// CHECK_NEW_DISPATCH-LABEL: define{{.*}} void @f0
// CHECK_NEW_DISPATCH: @_objc_msgSend_fixup_alloc
// CHECK_NEW_DISPATCH-LABEL: define{{.*}} void @f1
// CHECK_NEW_DISPATCH: load {{.*}}OBJC_SELECTOR_REFERENCES
//
// RUN: %clang_cc1 -fobjc-dispatch-method=legacy -emit-llvm -o - %s | FileCheck -check-prefix=CHECK_OLD_DISPATCH %s
//
// CHECK_OLD_DISPATCH-LABEL: define {{.*}}void @f0
// CHECK_OLD_DISPATCH: load {{.*}}OBJC_SELECTOR_REFERENCES
// CHECK_OLD_DISPATCH-LABEL: define {{.*}}void @f1
// CHECK_OLD_DISPATCH: load {{.*}}OBJC_SELECTOR_REFERENCES

@interface A
+(id) alloc;
-(int) im0;
@end

void f0(void) {
  [A alloc];
}

void f1(A *a) {
  [a im0];
}
