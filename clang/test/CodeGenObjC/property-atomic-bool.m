// RUN: %clang_cc1 -triple x86_64-apple-macosx10 -emit-llvm -x objective-c %s -o - | FileCheck %s

// CHECK: define internal zeroext i1 @"\01-[A0 p]"(
// CHECK:   %[[ATOMIC_LOAD:.*]] = load atomic i8, i8* %{{.*}} seq_cst
// CHECK:   %[[TOBOOL:.*]] = trunc i8 %[[ATOMIC_LOAD]] to i1
// CHECK:   ret i1 %[[TOBOOL]]

// CHECK: define internal void @"\01-[A0 setP:]"({{.*}} i1 zeroext {{.*}})
// CHECK:   store atomic i8 %{{.*}}, i8* %{{.*}} seq_cst
// CHECK:   ret void

// CHECK: define internal zeroext i1 @"\01-[A1 p]"(
// CHECK:   %[[ATOMIC_LOAD:.*]] = load atomic i8, i8* %{{.*}} unordered
// CHECK:   %[[TOBOOL:.*]] = trunc i8 %load to i1
// CHECK:   ret i1 %[[TOBOOL]]

// CHECK: define internal void @"\01-[A1 setP:]"({{.*}} i1 zeroext %p)
// CHECK:   store atomic i8 %{{.*}}, i8* %{{.*}} unordered
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
