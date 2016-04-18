// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -std=c++11 -ast-dump %s | FileCheck %s
// CHECK: CXXCtorInitializer Field {{.*}} 'ptr' 'void *'

@interface NSObject
@end

@interface I : NSObject
@end

struct S {
  S();
  void *ptr = nullptr;
};

@implementation I
S::S() {}
@end
