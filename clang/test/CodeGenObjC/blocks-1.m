// RUN: %clang_cc1 %s -emit-llvm -o %t -fobjc-gc -fblocks -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5
// RUN: grep "_Block_object_dispose" %t | count 6
// RUN: grep "__copy_helper_block_" %t | count 4
// RUN: grep "__destroy_helper_block_" %t | count 4
// RUN: grep "__Block_byref_object_copy_" %t | count 2
// RUN: grep "__Block_byref_object_dispose_" %t | count 2
// RUN: grep "i32 135)" %t | count 0
// RUN: grep "_Block_object_assign" %t | count 4
// RUN: grep "objc_read_weak" %t | count 2
// RUN: grep "objc_assign_weak" %t | count 3
// RUN: %clang_cc1 -x objective-c++ %s -emit-llvm -o %t -fobjc-gc -fblocks -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5
// RUN: grep "_Block_object_dispose" %t | count 6
// RUN: grep "__copy_helper_block_" %t | count 4
// RUN: grep "__destroy_helper_block_" %t | count 4
// RUN: grep "__Block_byref_object_copy_" %t | count 2
// RUN: grep "__Block_byref_object_dispose_" %t | count 2
// RUN: grep "i32 135)" %t | count 0
// RUN: grep "_Block_object_assign" %t | count 4
// RUN: grep "objc_read_weak" %t | count 2
// RUN: grep "objc_assign_weak" %t | count 3

@interface NSDictionary @end

void test1(NSDictionary * dict) {
  ^{ (void)dict; }();
}

@interface D
@end

void foo() {
  __block __weak D *weakSelf;
  D *l;
  l = weakSelf;
  weakSelf = l;
}

void (^__weak b)(void);

void test2() {
  __block int i = 0;
  b = ^ {  ++i; };
}
