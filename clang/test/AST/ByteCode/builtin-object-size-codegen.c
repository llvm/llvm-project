// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1                                         -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s


#define PS(N) __attribute__((pass_object_size(N)))
  int ObjectSize0(void *const p PS(0)) {
    return __builtin_object_size(p, 0);
  }

  int ObjectSize1(void *const p PS(1)) {
    return __builtin_object_size(p, 1);
  }

  int ObjectSize2(void *const p PS(2)) {
    return __builtin_object_size(p, 2);
  }

  int ObjectSize3(void *const p PS(3)) {
    return __builtin_object_size(p, 3);
  }

  struct Foo {
    int t[10];
  };


  int gi;
  void test1(unsigned long sz) {
    struct Foo t[10];

    // CHECK: call i32 @ObjectSize0(ptr noundef %{{.*}}, i64 noundef 360)
    gi = ObjectSize0(&t[1]);
    // call i32 @ObjectSize1(ptr noundef %{{.*}}, i64 noundef 360)
    // gi = ObjectSize2(&t[1]);
    // gi = ObjectSize2(&t[1].t[1]);
  }

/// Used to crash due to the void-typed ArraySubscriptExpr.
void foo(void *p) {
  int i = __builtin_object_size(&p[2], 3);
}
