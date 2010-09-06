// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

struct foo {
    void *a;
    int b;
};

// CHECK: @u = global %union.anon zeroinitializer
union { int i; float f; } u = { };

// CHECK: @u2 = global %1 { i32 0, [4 x i8] undef }
union { int i; double f; } u2 = { };

// CHECK: @u3 = global %2 zeroinitializer
union { double f; int i; } u3 = { };

// CHECK: @b = global [2 x i32] [i32 0, i32 22]
int b[2] = {
  [1] = 22
};

void test1(int argc, char **argv)
{
  // CHECK: internal global %struct.foo { i8* null, i32 1024 }
  static struct foo foo = {
    .b = 1024,
  };

  // CHECK: bitcast %union.anon* %u2
  // CHECK: call void @llvm.memset
   union { int i; float f; } u2 = { };

  // CHECK-NOT: call void @llvm.memset
  union { int i; float f; } u3;

  // CHECK: ret void
}


// PR7151
struct S {
  int nkeys;
  int *keys;
  union {
    void *data;
  };
};

void test2() {
  struct S *btkr;
  
  *btkr = (struct S) {
    .keys  = 0,
    { .data  = 0 },
  };
}
