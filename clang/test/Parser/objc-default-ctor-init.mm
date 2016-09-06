// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -std=c++11 -ast-dump %s | FileCheck %s
// CHECK: CXXCtorInitializer Field {{.*}} 'ptr' 'void *'
// CHECK: CXXCtorInitializer Field {{.*}} 'q' 'struct Q'

@interface NSObject
@end

@interface I : NSObject
@end

struct Q { Q(); };

struct S {
  S();
  void *ptr = nullptr;
  Q q;
};

@implementation I
S::S() {}
@end
