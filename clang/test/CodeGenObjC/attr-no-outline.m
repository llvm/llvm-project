// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - -DTEST_ATTR  | FileCheck %s --check-prefixes=OBJC,OBJC-ATTR
// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - -mno-outline | FileCheck %s --check-prefixes=OBJC,OBJC-ARG
// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-unknown-linux-gnu -o -              | FileCheck %s --check-prefixes=OBJC,OBJC-NONE

// RUN: %clang_cc1 -emit-llvm -x objective-c++ %s -triple x86_64-unknown-linux-gnu -o - -DTEST_ATTR  | FileCheck %s --check-prefixes=OBJCXX,OBJCXX-ATTR
// RUN: %clang_cc1 -emit-llvm -x objective-c++ %s -triple x86_64-unknown-linux-gnu -o - -mno-outline | FileCheck %s --check-prefixes=OBJCXX,OBJCXX-ARG
// RUN: %clang_cc1 -emit-llvm -x objective-c++ %s -triple x86_64-unknown-linux-gnu -o -              | FileCheck %s --check-prefixes=OBJCXX,OBJCXX-NONE

// This test checks that:
// - [[clang::no_outline]] adds the nooutline IR attribute to specific definitions
// - `-mno-outline` adds the nooutline IR attribute to all definitions
// - Lack of either does not add nooutline IR attribute


#ifdef TEST_ATTR
#define ATTR [[clang::no_outline]]
#else
#define ATTR
#endif

@interface Test
- (int)method:(int)x;
- (int)method_no_attr:(int)x;
+ (int)static_method:(int)x;
+ (int)static_method_no_attr:(int)x;
@end

@implementation Test

// OBJC-LABEL: define internal i32 @"\01-[Test method:]"(
// OBJC: ) #[[ATTR0:[0-9]+]] {

// OBJCXX-LABEL: define internal noundef i32 @"\01-[Test method:]"(
// OBJCXX: ) #[[ATTR0:[0-9]+]] {
- (int)method:(int)x ATTR {
  return x;
}

// OBJC-LABEL: define internal i32 @"\01-[Test method_no_attr:]"(
// OBJC-ATTR: ) #[[ATTR1:[0-9]+]] {
// OBJC-ARG:  ) #[[ATTR0]] {
// OBJC-NONE: ) #[[ATTR0]] {

// OBJCXX-LABEL: define internal noundef i32 @"\01-[Test method_no_attr:]"(
// OBJCXX-ATTR: ) #[[ATTR1:[0-9]+]] {
// OBJCXX-ARG:  ) #[[ATTR0]] {
// OBJCXX-NONE: ) #[[ATTR0]] {
- (int)method_no_attr:(int) x {
  return x;
}

// OBJC-LABEL: define internal i32 @"\01+[Test static_method:]"(
// OBJC: ) #[[ATTR0]] {

// OBJCXX-LABEL: define internal noundef i32 @"\01+[Test static_method:]"(
// OBJCXX: ) #[[ATTR0]] {
+ (int)static_method:(int)x ATTR {
  return x;
}


// OBJC-LABEL: define internal i32 @"\01+[Test static_method_no_attr:]"(
// OBJC-ATTR: ) #[[ATTR1]] {
// OBJC-ARG:  ) #[[ATTR0]] {
// OBJC-NONE: ) #[[ATTR0]] {


// OBJCXX-LABEL: define internal noundef i32 @"\01+[Test static_method_no_attr:]"(
// OBJCXX-ATTR: ) #[[ATTR1]] {
// OBJCXX-ARG:  ) #[[ATTR0]] {
// OBJCXX-NONE: ) #[[ATTR0]] {

+ (int)static_method_no_attr:(int)x {
  return x;
}

@end

// OBJC: attributes #[[ATTR0]] = {
// OBJC-ATTR-SAME: nooutline
// OBJC-ARG-SAME: nooutline
// OBJC-NONE-NOT: nooutline
// OBJC-SAME: }

// OBJC-ATTR: attributes #[[ATTR1]] = {
// OBJC-ATTR-NOT: nooutline
// OBJC-ATTR-SAME: }

// OBJCXX: attributes #[[ATTR0]] = {
// OBJCXX-ATTR-SAME: nooutline
// OBJCXX-ARG-SAME: nooutline
// OBJCXX-NONE-NOT: nooutline
// OBJCXX-SAME: }

// OBJCXX-ATTR: attributes #[[ATTR1]] = {
// OBJCXX-ATTR-NOT: nooutline
// OBJCXX-ATTR-SAME: }
