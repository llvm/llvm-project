// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -fblocks -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define{{.*}} void @_Z28test_objc_object_pseudo_dtorPU8__strongP11objc_objectPU6__weakS0_
void test_objc_object_pseudo_dtor(__strong id *ptr, __weak id *wptr) {
  // CHECK: load i8**, i8***
  // CHECK-NEXT: load i8*, i8**
  // CHECK-NEXT: call void @llvm.objc.release
  ptr->~id();

  // CHECK: call void @llvm.objc.destroyWeak(i8** {{%.*}})
  wptr->~id();

  // CHECK: load i8**, i8***
  // CHECK-NEXT: load i8*, i8**
  // CHECK-NEXT: call void @llvm.objc.release
  (*ptr).~id();

  // CHECK: call void @llvm.objc.destroyWeak(i8** {{%.*}})
  (*wptr).~id();
  // CHECK: ret void
}
