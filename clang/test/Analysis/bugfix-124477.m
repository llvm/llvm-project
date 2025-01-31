// RUN: %clang_analyze_cc1 -analyzer-checker=core,apiModeling,nullability.NullableDereferenced,nullability.NullabilityBase -x objective-c %s
/*
  This test is reduced from a static analyzer crash. The bug causing
  the crash is explained in #124477.  It can only be triggered in some
  rare cases so please do not modify this reproducer.
*/

#pragma clang assume_nonnull begin
# 15 "some-sys-header.h" 1 3
@class NSArray, NSObject;

@interface Base
@property (readonly, copy) NSArray *array;
@end

#pragma clang assume_nonnull end
# 8 "this-file.m" 2


@interface Test : Base

@property (readwrite, copy, nullable) NSObject *label;
@property (readwrite, strong, nullable) Test * field;

- (void)f;

@end

@implementation Test
- (void)f
{
  NSObject * X;

  for (NSObject *ele in self.field.array) {}
  self.label = X;  
}
@end


