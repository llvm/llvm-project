// Test that the use of thread local variables would be wrapped by @llvm.threadlocal.address intrinsics.
// RUN: %clang_cc1 -std=c++11 -emit-llvm -triple x86_64 -o - %s -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm -triple aarch64 -o - -O1 %s | FileCheck %s -check-prefix=CHECK-O1
thread_local int i;
int g() {
  i++;
  return i;
}
// CHECK: @i = {{.*}}thread_local global i32 0
// CHECK: @_ZZ1fvE1j = internal thread_local global i32 0
//
// CHECK: @_Z1gv()
// CHECK-NEXT: entry
// CHECK-NEXT: %[[IA:.+]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @i)
// CHECK-NEXT: %[[VA:.+]] = load i32, ptr %[[IA]]
// CHECK-NEXT: %[[INC:.+]] = add nsw i32 %[[VA]], 1
// CHECK-NEXT: store i32 %[[INC]], ptr %[[IA]], align 4
// CHECK-NEXT: %[[IA2:.+]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @i)
// CHECK-NEXT: %[[RET:.+]] = load i32, ptr %[[IA2]], align 4
// CHECK-NEXT: ret i32 %[[RET]]
//
// CHECK: declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #[[ATTR_NUM:.+]]
//
// CHECK-O1-LABEL: @_Z1gv
// CHECK-O1-NEXT: entry:
// CHECK-O1-NEXT:   %[[I_ADDR:.+]] = {{.*}}call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @i)
// CHECK-O1-NEXT:   %[[VAL:.+]] = load i32, ptr %[[I_ADDR]]
// CHECK-O1-NEXT:   %[[INC:.+]] = add nsw i32 %[[VAL]], 1
// CHECK-O1-NEXT:   store i32 %[[INC]], ptr %[[I_ADDR]]
// CHECK-O1-NEXT:   ret i32 %[[INC]]
int f() {
  thread_local int j = 0;
  j++;
  return j;
}
// CHECK: @_Z1fv()
// CHECK-NEXT: entry
// CHECK-NEXT: %[[JA:.+]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @_ZZ1fvE1j)
// CHECK-NEXT: %[[VA:.+]] = load i32, ptr %[[JA]]
// CHECK-NEXT: %[[INC:.+]] = add nsw i32 %[[VA]], 1
// CHECK-NEXT: store i32 %[[INC]], ptr %[[JA]], align 4
// CHECK-NEXT: %[[JA2:.+]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @_ZZ1fvE1j)
// CHECK-NEXT: %[[RET:.+]] = load i32, ptr %[[JA2]], align 4
// CHECK-NEXT: ret i32 %[[RET]]
//
// CHECK-O1-LABEL: @_Z1fv
// CHECK-O1-NEXT: entry:
// CHECK-O1-NEXT:   %[[J_ADDR:.+]] = {{.*}}call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @_ZZ1fvE1j)
// CHECK-O1-NEXT:   %[[VAL:.+]] = load i32, ptr %[[J_ADDR]]
// CHECK-O1-NEXT:   %[[INC:.+]] = add nsw i32 %[[VAL]], 1
// CHECK-O1-NEXT:   store i32 %[[INC]], ptr %[[J_ADDR]]
// CHECK-O1-NEXT:   ret i32 %[[INC]]
//
// CHECK: attributes #[[ATTR_NUM]] = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
