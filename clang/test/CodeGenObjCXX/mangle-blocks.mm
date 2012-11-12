// RUN: %clang_cc1 -emit-llvm -fblocks -o - -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck %s

// CHECK: @_ZGVN3foo22___Z3foov_block_invoke5valueE = internal global i64 0

int f();

void foo() {
  // CHECK: define internal i32 @___Z3foov_block_invoke
  // CHECK: call i32 @__cxa_guard_acquire(i64* @_ZGVN3foo22___Z3foov_block_invoke5valueE
  (void)^(int x) { 
    static int value = f();
    return x + value;
  };
}

// CHECK: define internal i32 @i_block_invoke
int i = ^(int x) { return x;}(i);

@interface A
- (void)method;
@end

@implementation A
- (void)method { 
  // CHECK: define internal signext i8 @"__11-[A method]_block_invoke"
  (void)^(int x) {
    // CHECK: @"_ZN11-[A method]28__11-[A method]_block_invoke4nameE"
    static const char *name = "hello";
    return name[x];
  };
}
@end

void foo(int) {
  (void)^(int x) { 
    static const char *name = "hello";
    return name[x];
  };
}

namespace N {
  // CHECK: define internal signext i8 @___Z3fooi_block_invoke
  void bar() {
    (void)^(int x) { 
      // CHECK: @_ZN1N3bar26___ZN1N3barEv_block_invoke4nameE
      static const char *name = "hello";
      return name[x];
    };
  }
}
