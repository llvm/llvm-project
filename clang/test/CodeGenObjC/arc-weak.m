// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck %s

__attribute((objc_root_class)) @interface A @end
@interface B : A @end

// rdar://problem/23559789
//   Ensure that type differences don't cause an assert here.
void test0(__weak B **src) {
  __weak A *dest = *src;
}
// CHECK-LABEL: define{{.*}} void @test0
// CHECK:       [[SRC:%.*]] = alloca [[B:%.*]]**, align 8
// CHECK:       [[DEST:%.*]] = alloca [[A:%.*]]*, align 8
// CHECK:       [[T0:%.*]] = load [[B]]**, [[B]]*** [[SRC]], align 8
// CHECK-NEXT:  [[T1:%.*]] = bitcast [[B]]** [[T0]] to [[A]]**
// CHECK-NEXT:  [[T2:%.*]] = bitcast [[A]]** [[DEST]] to i8**
// CHECK-NEXT:  [[T3:%.*]] = bitcast [[A]]** [[T1]] to i8**
// CHECK-NEXT:  call void @llvm.objc.copyWeak(i8** [[T2]], i8** [[T3]])
// CHECK-NEXT:  [[T0:%.*]] = bitcast [[A]]** [[DEST]] to i8**
// CHECK:       call void @llvm.objc.destroyWeak(i8** [[T0]])
