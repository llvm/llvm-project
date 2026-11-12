// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 %s -emit-llvm -o - | FileCheck %s

// CHECK: @"__func__.-[Foo instanceTest1]" = private unnamed_addr constant [21 x i8] c"-[Foo instanceTest1]\00"
// CHECK: @"__func__.-[Foo instanceTest2:]" = private unnamed_addr constant [22 x i8] c"-[Foo instanceTest2:]\00"
// CHECK: @"__func__.-[Foo instanceTest3:withB:]" = private unnamed_addr constant [28 x i8] c"-[Foo instanceTest3:withB:]\00"
// CHECK: @"__func__.-[Foo instanceTest4]" = private unnamed_addr constant [21 x i8] c"-[Foo instanceTest4]\00"
// CHECK: @"__func__.+[Foo classTest1]" = private unnamed_addr constant [18 x i8] c"+[Foo classTest1]\00"
// CHECK: @"__func__.+[Foo classTest2:]" = private unnamed_addr constant [19 x i8] c"+[Foo classTest2:]\00"
// CHECK: @"__func__.+[Foo classTest3:withB:]" = private unnamed_addr constant [25 x i8] c"+[Foo classTest3:withB:]\00"
// CHECK: @"__func__.+[Foo classTest4]" = private unnamed_addr constant [18 x i8] c"+[Foo classTest4]\00"
// CHECK: @"__func__.-[Foo(Category) instanceTestWithCategory]" = private unnamed_addr constant [42 x i8] c"-[Foo(Category) instanceTestWithCategory]\00"
// CHECK: @"__func__.+[Foo(Category) classTestWithCategory]" = private unnamed_addr constant [39 x i8] c"+[Foo(Category) classTestWithCategory]\00"

int printf(const char * _Format, ...);

@interface Foo
@end

@implementation Foo

- (void)instanceTest1 {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

- (void)instanceTest2:(int)i {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

- (void)instanceTest3:(int)a withB:(double)b {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

- (int)instanceTest4 {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
  return 0;
}

+ (void)classTest1 {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

+ (void)classTest2:(int)i {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

+ (void)classTest3:(int)a withB:(double)b {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

+ (int)classTest4 {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
  return 0;
}

@end

@interface Foo (Category)
@end

@implementation Foo (Category)

- (void)instanceTestWithCategory {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

+ (void)classTestWithCategory {
  printf("__func__: %s\n", __func__);
  printf("__FUNCTION__: %s\n", __FUNCTION__);
  printf("__PRETTY_FUNCTION__: %s\n\n", __PRETTY_FUNCTION__);
}

@end
