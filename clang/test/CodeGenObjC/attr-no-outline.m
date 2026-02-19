// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s --check-prefix=OBJC
// RUN: %clang_cc1 -emit-llvm -x objective-c++ %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s --check-prefix=OBJCXX

@interface Test
- (int)method:(int)x;
+ (int)static_method:(int)x;
@end

@implementation Test

// OBJC-LABEL: define internal i32 @"\01-[Test method:]"(
// OBJC: ) #[[ATTR0:[0-9]+]] {

// OBJCXX-LABEL: define internal noundef i32 @"\01-[Test method:]"(
// OBJCXX: ) #[[ATTR0:[0-9]+]] {
- (int)method:(int)x [[clang::no_outline]] {
  return x;
}

// OBJC-LABEL: define internal i32 @"\01+[Test static_method:]"(
// OBJC: ) #[[ATTR0]] {

// OBJCXX-LABEL: define internal noundef i32 @"\01+[Test static_method:]"(
// OBJCXX: ) #[[ATTR0]] {
+ (int)static_method:(int)x [[clang::no_outline]] {
  return x;
}

@end

// OBJC: attributes #[[ATTR0]] = {
// OBJC-SAME: nooutline
// OBJC-SAME: }

// OBJCXX: attributes #[[ATTR0]] = {
// OBJCXX-SAME: nooutline
// OBJCXX-SAME: }



