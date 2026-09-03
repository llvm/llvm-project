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

void foo2(struct Foo *t) {
  // CHECK: call i32 @ObjectSize3(ptr noundef %{{.*}}, i64 noundef 36)
  ObjectSize3(&t->t[1]);
}


/// Used to crash due to the void-typed ArraySubscriptExpr.
void foo(void *p) {
  int i = __builtin_object_size(&p[2], 3);
}

struct DynStructVar {
  char fst[16];
  char snd[];
};

static struct DynStructVar D32 = {
  .fst = {},
  .snd = { 0, 1, 2, 3, 4, 5, 6 },
};

// CHECK-LABEL: @test32
void test32(void) {
  // CHECK: store i32 23
  gi = __builtin_object_size(&D32, 0);
  // CHECK: store i32 23
  gi = __builtin_object_size(&D32, 1);
  // CHECK: store i32 23
  gi = __builtin_object_size(&D32, 2);
  // CHECK: store i32 23
  gi = __builtin_object_size(&D32, 3);

  // CHECK: store i32 7
  gi = __builtin_object_size(&D32.snd[0], 0);
  // CHECK: store i32 1
  gi = __builtin_object_size(&D32.snd[6], 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(&D32.snd[10], 0);
}

struct S {
  char c[7];
  char k[];
};

struct S s = {
  .c = {1,2,3,4,5,6,7},
  .k = {1,2,3,4,5    }
};

// CHECK-LABEL: @testflex
void testflex() {
  int gi;
  // CHECK: store i32 5
  gi = __builtin_object_size(&s.k, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&s.k, 1);
  // CHECK: store i32 5
  gi = __builtin_object_size(&s.k, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(&s.k, 3);

  // CHECK: store i32 2
  gi = __builtin_object_size(&s.k[3], 0);
  // CHECK: store i32 2
  gi = __builtin_object_size(&s.k[3], 1);
  // CHECK: store i32 2
  gi = __builtin_object_size(&s.k[3], 2);
  /// The following fails to evaluate in clang but returns 2 in GCC.
  // store i32 0
  gi = __builtin_object_size(&s.k[3], 3);
}

// CHECK-LABEL: @vlas
void vlas(int size) {
  char z[size];

  int gi;
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(z, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(z, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(z, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(z, 3);
}

struct hh {
  char s1[0];
  char * s2;
};
// CHECK-LABEL: @f17
void f17(void) {
  struct hh h0;
  int gi;
  // CHECK: store i32 8
  gi = __builtin_object_size(h0.s1, 0);
}
