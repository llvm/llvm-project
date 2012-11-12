// RUN: %clang_cc1 -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t1 %s
// RUN: FileCheck -check-prefix=CHECK-I386 < %t1 %s

// RUN: %clang_cc1 -triple armv6-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -target-abi apcs-gnu -emit-llvm -o %t2 %s
// RUN: FileCheck -check-prefix=CHECK-ARM < %t2 %s

@interface I0 { 
@public
    unsigned x:15;
    unsigned y: 1;
} 
@end

// Check that we don't try to use an i32 load here, which would reach beyond the
// end of the structure.
//
// CHECK-I386: define i32 @f0(
// CHECK-I386:   [[t0_0:%.*]] = load i16* {{.*}}, align 1
// CHECK-I386:   lshr i16 [[t0_0]], 7
// CHECK-I386: }
int f0(I0 *a) {
  return a->y;
}

// Check that we can handled straddled loads.
//
// CHECK-ARM: define i32 @f1(
// CHECK-ARM:    [[t1_ptr:%.*]] = getelementptr
// CHECK-ARM:    [[t1_base:%.*]] = bitcast i8* [[t1_ptr]] to i32*
// CHECK-ARM:    [[t1_0:%.*]] = load i32* [[t1_base]], align 1
// CHECK-ARM:    lshr i32 [[t1_0]], 1
// CHECK-ARM:    [[t1_base_2_cast:%.*]] = bitcast i32* %{{.*}} to i8*
// CHECK-ARM:    [[t1_base_2:%.*]] = getelementptr i8* [[t1_base_2_cast]]
// CHECK-ARM:    [[t1_1:%.*]] = load i8* [[t1_base_2]], align 1
// CHECK-ARM:    and i8 [[t1_1:%.*]], 1
// CHECK-ARM: }
@interface I1 {
@public
    unsigned x: 1;
    unsigned y:32;
}
@end

int f1(I1 *a) { return a->y; }
