// RUN: %clang_cc1 -emit-llvm -fblocks -o - -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck %s

// CHECK-DAG: @_ZZZN26externally_visible_statics1S3fooEiEd_Ub_E1k = linkonce_odr global i32 0
// CHECK-DAG: @_ZZZN26externally_visible_statics10inlinefuncEvEUb_E1i = linkonce_odr global i32 0
// CHECK-DAG: @_ZZZN26externally_visible_statics10inlinefuncEvEUb0_E1j = linkonce_odr global i32 0
// CHECK-DAG: @_ZZ26externally_visible_statics1S1xMUb_E1j = linkonce_odr global i32 0

int f();

void foo() {
  // CHECK-LABEL: define internal i32 @___Z3foov_block_invoke
  // CHECK: call i32 @__cxa_guard_acquire(i64* @_ZGVZZ3foovEUb_E5value
  (void)^(int x) { 
    static int value = f();
    return x + value;
  };
}

// CHECK-LABEL: define internal i32 @i_block_invoke
int i = ^(int x) { return x;}(i);

@interface A
- (void)method;
@end

@implementation A
- (void)method { 
  // CHECK: define internal signext i8 @"__11-[A method]_block_invoke"
  (void)^(int x) {
    // CHECK: @"_ZZZ11-[A method]EUb1_E4name"
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
  // CHECK-LABEL: define internal signext i8 @___Z3fooi_block_invoke
  void bar() {
    (void)^(int x) { 
      // CHECK: @_ZZZN1N3barEvEUb3_E4name
      static const char *name = "hello";
      return name[x];
    };
  }
}

class C {
  C();
};
C::C() {
  (void)^(int x) { 
    // CHECK: @_ZZZN1CC1EvEUb4_E5nameb
    static const char *nameb = "hello";
    return nameb[x];
  };
}

int f();
namespace externally_visible_statics {
  inline void inlinefunc() {
    ^{
      static int i = f();
    }();
    ^{
      static int j = f();
    }();
  }
  struct S {
    int x = ^{
      static int j = f();
      return j;
    }();
    void foo(int y = ^{ static int k = f(); return k; }()) {}
  };
  void g() {
    inlinefunc();
    S s;
    s.foo();
  }
}
