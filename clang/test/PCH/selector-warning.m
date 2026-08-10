// RUN: %clang_cc1 -x objective-c -emit-pch -o %t.h.pch %S/selector-warning.h
// RUN: %clang_cc1 -include-pch %t.h.pch %s

@interface Bar 
+ (void) clNotOk;
- (void) instNotOk;
+ (void) cl1;
@end

@implementation Bar
- (void) bar {}
+ (void) cl1 {}
+ (void) cl2 {}
@end

@implementation Bar(CAT)
- (void) b1ar {}
@end

