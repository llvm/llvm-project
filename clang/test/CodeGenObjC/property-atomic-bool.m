// RUN: %clang_cc1 -triple x86_64-apple-macosx10 -emit-llvm -x objective-c %s -o - | FileCheck %s

// CHECK-LABEL: define internal zeroext i1 @"\01-[A0 p]"(
// CHECK:   %[[ATOMIC_LOAD:.*]] = load atomic i8, ptr %{{.*}} seq_cst, align 1
// CHECK:   %[[TOBOOL:.*]] = icmp ne i8 %[[ATOMIC_LOAD]], 0
// CHECK:   ret i1 %[[TOBOOL]]

// CHECK-LABEL: define internal void @"\01-[A0 setP:]"({{.*}} i1 noundef zeroext {{.*}})
// CHECK:   store atomic i8 %{{.*}}, ptr %{{.*}} seq_cst, align 1
// CHECK:   ret void

// CHECK-LABEL: define internal zeroext i1 @"\01-[A1 p]"(
// CHECK:   %[[ATOMIC_LOAD:.*]] = load atomic i8, ptr %{{.*}} unordered, align 1
// CHECK:   %[[TOBOOL:.*]] = icmp ne i8 %[[ATOMIC_LOAD]], 0
// CHECK:   ret i1 %[[TOBOOL]]

// CHECK-LABEL: define internal void @"\01-[A1 setP:]"({{.*}} i1 noundef zeroext %p)
// CHECK:   store atomic i8 %{{.*}}, ptr %{{.*}} unordered, align 1
// CHECK:   ret void

@interface A0
@property(nonatomic) _Atomic(_Bool) p;
@end
@implementation A0
@end

@interface A1 {
  _Atomic(_Bool) p;
}
@property _Atomic(_Bool) p;
@end
@implementation A1
@synthesize p;
@end
