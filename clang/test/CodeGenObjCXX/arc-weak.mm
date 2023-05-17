// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -std=c++11 -o - %s | FileCheck %s

__attribute((objc_root_class)) @interface A @end
@interface B : A @end

// rdar://problem/23559789
//   Ensure that type differences don't cause an assert here.
void test0(__weak B **src) {
  __weak A *dest = *src;
}
// CHECK-LABEL: define{{.*}} void @_Z5test0PU6__weakP1B(
// CHECK:       [[SRC:%.*]] = alloca ptr, align 8
// CHECK:       [[DEST:%.*]] = alloca ptr, align 8
// CHECK:       [[T0:%.*]] = load ptr, ptr [[SRC]], align 8
// CHECK-NEXT:  call void @llvm.objc.copyWeak(ptr [[DEST]], ptr [[T0]])
// CHECK:       call void @llvm.objc.destroyWeak(ptr [[DEST]])

void test1(__weak B **src) {
  __weak A *dest = static_cast<__weak B*&&>(*src);
}
// CHECK-LABEL: define{{.*}} void @_Z5test1PU6__weakP1B(
// CHECK:       [[SRC:%.*]] = alloca ptr, align 8
// CHECK:       [[DEST:%.*]] = alloca ptr, align 8
// CHECK:       [[T0:%.*]] = load ptr, ptr [[SRC]], align 8
// CHECK-NEXT:  call void @llvm.objc.moveWeak(ptr [[DEST]], ptr [[T0]])
// CHECK:       call void @llvm.objc.destroyWeak(ptr [[DEST]])
