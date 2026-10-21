// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1                                         -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @_Z3foov
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

  C c2{};
  // CHECK: store i32 16
  gi = __builtin_object_size(&c2.bs[0], 1);
}

// CHECK-LABEL: @_Z4foo2v
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

// CHECK-LABEL: @_Z6babar0P9foofoo0_t
unsigned babar0(foofoo0_t *f) {
  // CHECK: ret i32 0
  return __builtin_object_size(f->c, 1);
}

// CHECK-LABEL: @_Z5test2v
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

// CHECK-LABEL: @_Z5test3v
void test3() {
  struct A {
    int a;
  };
  struct B {
    int b;
  };
  struct C : A, B {};

  C c;

  int gi;
  // CHECK: store i32 8
  gi = __builtin_object_size((B*)&c, 3);

}

struct A { char buf[16]; };
struct B : A {};
struct C { int i; B bs[1]; } *c;
// CHECK-LABEL: @_Z13globalPointerv
void globalPointer() {
  int gi;
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(&c->bs[0], 2);
}

// CHECK-LABEL: @_Z11nonPtrParam1C
void nonPtrParam(C c) {
  int gi;
  // CHECK: store i32 16
  gi = __builtin_object_size(&c.bs[0], 2);
}


struct X {
  char p[7];
};

struct Y: X {
  char p[3];
};

struct F {
  Y y;
};

// CHECK-LABEL: @_Z6testXYv
void testXY() {
  int gi;
  Y y;

  // CHECK: store i32 10
  gi = __builtin_object_size(&y, 0);
  // CHECK: store i32 10
  gi = __builtin_object_size(&y, 1);
  // CHECK: store i32 10
  gi = __builtin_object_size(&y, 2);
  // CHECK: store i32 10
  gi = __builtin_object_size(&y, 3);

  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&y, 0);
  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&y, 1);
  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&y, 2);
  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&y, 3);


  F f;
  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&f.y, 0);
  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&f.y, 1);
  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&f.y, 2);
  // CHECK: store i32 10
  gi = __builtin_object_size((X*)&f.y, 3);


  // CHECK: store i32 6
  gi = __builtin_object_size(&((X*)&f.y)->p[4], 0);
  // CHECK: store i32 3
  gi = __builtin_object_size(&((X*)&f.y)->p[4], 1);
  // CHECK: store i32 6
  gi = __builtin_object_size(&((X*)&f.y)->p[4], 2);
  // CHECK: store i32 3
  gi = __builtin_object_size(&((X*)&f.y)->p[4], 3);
}

// CHECK-LABEL: @_Z7testOPEv
int s;
void testOPE() {
  int gi;

  // CHECK: store i32 4
  gi = __builtin_object_size(&s, 0);
  // CHECK: store i32 4
  gi = __builtin_object_size(&s, 1);
  // CHECK: store i32 4
  gi = __builtin_object_size(&s, 2);
  // CHECK: store i32 4
  gi = __builtin_object_size(&s, 3);

  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 1, 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 1, 1);
  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 1, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 1, 3);

  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 20, 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 20, 1);
  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 20, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(&s + 20, 3);
}

struct K {char p[6]; };
// CHECK-LABEL: @_Z18testArrayAddOffsetv
void testArrayAddOffset() {
  int gi;

  K ks[4];
  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 1, 0);
  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 1, 1);
  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 1, 2);
  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 1, 3);

  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 3 - 2, 0);
  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 3 - 2, 1);
  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 3 - 2, 2);
  // CHECK: store i32 18
  gi = __builtin_object_size(ks + 3 - 2, 3);

  // CHECK: store i32 0
  gi = __builtin_object_size(ks - 5, 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(ks - 5, 1);
  // CHECK: store i32 0
  gi = __builtin_object_size(ks - 5, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(ks - 5, 3);
}


struct LoadCommandInfo {
  char *Ptr;
  int a;
  int b;
};

// CHECK-LABEL: @_Z16testNonConstBasev
void testNonConstBase() {
  struct A { char buf[16]; };
  struct B : A {};
  struct C { int i; B bs[1]; } *c;

  LoadCommandInfo LC;
  int gi;
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(LC.Ptr, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(LC.Ptr + 8, 0);
}


struct Ref_struct {
  int RD, Sib;
  int *Op;
};

struct NodeBase {
  int Next;
  Ref_struct RefData;
};

struct NodeAddr {
  NodeBase *Addr;
  int Id;
};

// CHECK-LABEL: @_Z9cloneNode8NodeAddr
void cloneNode(const NodeAddr B) {
  NodeBase NA_0;
  // memcpy(&NA_0, B.Addr, sizeof(NodeBase));

  int gi;

  // CHECK: store i32 24
  gi = __builtin_object_size(&NA_0, 0);
  // CHECK: store i32 24
  gi = __builtin_object_size(&NA_0, 1);
  // CHECK: store i32 24
  gi = __builtin_object_size(&NA_0, 2);
  // CHECK: store i32 24
  gi = __builtin_object_size(&NA_0, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(B.Addr, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(B.Addr, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 false)
  gi = __builtin_object_size(B.Addr, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(B.Addr, 3);
}
