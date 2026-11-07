// RUN: %clang_cc1 -triple arm64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -O0 -emit-llvm -fobjc-arc -fsanitize=thread -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx11.0.0 -fobjc-runtime=macosx-11.0.0 -O1 -emit-llvm -fobjc-arc -fsanitize=thread -o - %s | FileCheck %s

__attribute__((objc_root_class))
@interface CopyContainer {
  Class _isa;
}
@property(nonatomic, copy) id nonatomicCopyProp;
@property(atomic, copy) id atomicCopyProp;
@end

// CHECK-LABEL: define internal void @"\01-[CopyContainer setNonatomicCopyProp:]"(
// CHECK: [[ADDR:%.*]] = getelementptr i8, ptr [[SELF:%.*]], i64 {{.*}}
// CHECK-NEXT: call void @__tsan_write8(ptr [[ADDR]])
// CHECK-NEXT: {{(tail )?}}call void @objc_setProperty_nonatomic_copy(ptr noundef [[SELF]], ptr noundef {{.*}}, ptr noundef {{.*}}, i64 noundef {{.*}})

// CHECK-LABEL: define internal void @"\01-[CopyContainer setAtomicCopyProp:]"(
// CHECK-NOT: call void @__tsan_write8
// CHECK: {{(tail )?}}call void @objc_setProperty_atomic_copy(
@implementation CopyContainer
@end
