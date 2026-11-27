// RUN: %clang_cc1 -emit-llvm -fblocks -debug-info-kind=limited \
// RUN:   -fsanitize=nullability-return %s -o - | FileCheck %s

// Check that santizer check calls have a !dbg location.
// CHECK: call void {{.*}}@__ubsan_handle_nullability_return_v1_abort
// CHECK-SAME: !dbg

@protocol NSObject
@end

@interface NSObject<NSObject> {}
@end

#pragma clang assume_nonnull begin
@interface NSString : NSObject
+ (instancetype)stringWithFormat:(NSString *)format, ...;
@end

@interface NSIndexPath : NSObject {}
@end
#pragma clang assume_nonnull end

@interface B : NSObject
@end
id foo(NSIndexPath *indexPath) {
  return [B withBlock:^{
    return [NSString stringWithFormat:@"%ld",
                                      (long)[indexPath indexAtPosition:1]];
  }];
}
