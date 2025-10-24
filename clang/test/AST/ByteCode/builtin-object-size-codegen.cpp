// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1                                         -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

void foo() {
  struct A { char buf[16]; };
  struct B : A {};
  struct C { int i; B bs[1]; } *c;

  int gi;
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 2);
  // CHECK: store i32 16
  gi = __builtin_object_size(&c->bs[0], 3);
}


void foo2() {
  struct A { int a; };
  struct B { int b; };
  struct C: public A, public B {};

  C c;

  int gi;
  // CHECK: store i32 8
  gi = __builtin_object_size(&c, 0);
  // CHECK: store i32 8
  gi = __builtin_object_size((A*)&c, 0);
  // CHECK: store i32 4
  gi = __builtin_object_size((B*)&c, 0);

  // CHECK: store i32 8
  gi = __builtin_object_size((char*)&c, 0);
  // CHECK: store i32 8
  gi = __builtin_object_size((char*)(A*)&c, 0);
  // CHECK: store i32 4
  gi = __builtin_object_size((char*)(B*)&c, 0);
}


typedef struct {
  double c[0];
  float f;
} foofoo0_t;

unsigned babar0(foofoo0_t *f) {
  // CHECK: ret i32 0
  return __builtin_object_size(f->c, 1);
}

void test2() {
  struct A { char buf[16]; };
  struct B : A {};
  struct C { int i; B bs[1]; } *c;

  int gi;
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 2);
  // CHECK: store i32 16
  gi = __builtin_object_size(&c->bs[0], 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size((A*)&c->bs[0], 0);
  // CHECK: store i32 16
  gi = __builtin_object_size((A*)&c->bs[0], 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0].buf[0], 2);
  // CHECK: store i32 16
  gi = __builtin_object_size(&c->bs[0].buf[0], 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0].buf[0], 0);
  // CHECK: store i32 16
  gi = __builtin_object_size(&c->bs[0].buf[0], 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0].buf[0], 2);
  // CHECK: store i32 16
  gi = __builtin_object_size(&c->bs[0].buf[0], 3);
}
