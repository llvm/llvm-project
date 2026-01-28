// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

char gbuf[63];
char *gp;
int gi, gj;

// CIR-LABEL: @test1
// LLVM-LABEL: define {{.*}} void @test1
// OGCG-LABEL: define {{.*}} void @test1
void test1(void) {
  // CIR: cir.const #cir.int<59>
  // LLVM: store i32 59
  // OGCG: store i32 59
  gi = __builtin_object_size(&gbuf[4], 1);
}

// CIR-LABEL: @test2
// LLVM-LABEL: define {{.*}} void @test2
// OGCG-LABEL: define {{.*}} void @test2
void test2(void) {
  // CIR: cir.const #cir.int<63>
  // LLVM: store i32 63
  // OGCG: store i32 63
  gi = __builtin_object_size(gbuf, 1);
}

// CIR-LABEL: @test3
// LLVM-LABEL: define {{.*}} void @test3
// OGCG-LABEL: define {{.*}} void @test3
void test3(void) {
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&gbuf[100], 1);
}

// CIR-LABEL: @test4
// LLVM-LABEL: define {{.*}} void @test4
// OGCG-LABEL: define {{.*}} void @test4
void test4(void) {
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)(void*)&gbuf[-1], 1);
}

// CIR-LABEL: @test5
// LLVM-LABEL: define {{.*}} void @test5
// OGCG-LABEL: define {{.*}} void @test5
void test5(void) {
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(gp, 0);
}

// CIR-LABEL: @test6
// LLVM-LABEL: define {{.*}} void @test6
// OGCG-LABEL: define {{.*}} void @test6
void test6(void) {
  char buf[57];

  // CIR: cir.const #cir.int<53>
  // LLVM: store i32 53
  // OGCG: store i32 53
  gi = __builtin_object_size(&buf[4], 1);
}

// CIR-LABEL: @test18
// LLVM-LABEL: define {{.*}} i32 @test18
// OGCG-LABEL: define {{.*}} i32 @test18
unsigned test18(int cond) {
  int a[4], b[4];
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64
  // OGCG: call i64 @llvm.objectsize.i64
  return __builtin_object_size(cond ? a : b, 0);
}

// CIR-LABEL: @test19
// LLVM-LABEL: define {{.*}} void @test19
// OGCG-LABEL: define {{.*}} void @test19
void test19(void) {
  struct {
    int a, b;
  } foo;

  // CIR: cir.const #cir.int<8>
  // LLVM: store i32 8
  // OGCG: store i32 8
  gi = __builtin_object_size(&foo.a, 0);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&foo.a, 1);
  
  // CIR: cir.const #cir.int<8>
  // LLVM: store i32 8
  // OGCG: store i32 8
  gi = __builtin_object_size(&foo.a, 2);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&foo.a, 3);

  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&foo.b, 0);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&foo.b, 1);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&foo.b, 2);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&foo.b, 3);
}

// CIR-LABEL: @test20
// LLVM-LABEL: define {{.*}} void @test20
// OGCG-LABEL: define {{.*}} void @test20
void test20(void) {
  struct { int t[10]; } t[10];

  // CIR: cir.const #cir.int<380>
  // LLVM: store i32 380
  // OGCG: store i32 380
  gi = __builtin_object_size(&t[0].t[5], 0);
  
  // CIR: cir.const #cir.int<20>
  // LLVM: store i32 20
  // OGCG: store i32 20
  gi = __builtin_object_size(&t[0].t[5], 1);
  
  // CIR: cir.const #cir.int<380>
  // LLVM: store i32 380
  // OGCG: store i32 380
  gi = __builtin_object_size(&t[0].t[5], 2);
  
  // CIR: cir.const #cir.int<20>
  // LLVM: store i32 20
  // OGCG: store i32 20
  gi = __builtin_object_size(&t[0].t[5], 3);
}

// CIR-LABEL: @test21
// LLVM-LABEL: define {{.*}} void @test21
// OGCG-LABEL: define {{.*}} void @test21
void test21(void) {
  struct { int t; } t;

  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t + 1, 0);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t + 1, 1);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t + 1, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t + 1, 3);

  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t.t + 1, 0);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t.t + 1, 1);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t.t + 1, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t.t + 1, 3);
}

// CIR-LABEL: @test22
// LLVM-LABEL: define {{.*}} void @test22
// OGCG-LABEL: define {{.*}} void @test22
void test22(void) {
  struct { int t[10]; } t[10];

  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[10], 0);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[10], 1);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[10], 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[10], 3);

  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[9].t[10], 0);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[9].t[10], 1);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[9].t[10], 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[9].t[10], 3);

  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[0] + sizeof(t), 0);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[0] + sizeof(t), 1);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[0] + sizeof(t), 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[0] + sizeof(t), 3);

  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[9].t[0] + 10*sizeof(t[0].t), 0);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[9].t[0] + 10*sizeof(t[0].t), 1);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[9].t[0] + 10*sizeof(t[0].t), 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((char*)&t[9].t[0] + 10*sizeof(t[0].t), 3);
}

struct Test23Ty { int a; int t[10]; };

// CIR-LABEL: @test23
// LLVM-LABEL: define {{.*}} void @test23
// OGCG-LABEL: define {{.*}} void @test23
void test23(struct Test23Ty *p) {
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(p, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(p, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(p, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(p, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(&p->a, 0);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&p->a, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(&p->a, 2);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&p->a, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(&p->t[5], 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(&p->t[5], 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(&p->t[5], 2);
  
  // CIR: cir.const #cir.int<20>
  // LLVM: store i32 20
  // OGCG: store i32 20
  gi = __builtin_object_size(&p->t[5], 3);
}

// CIR-LABEL: @test24
// LLVM-LABEL: define {{.*}} void @test24
// OGCG-LABEL: define {{.*}} void @test24
void test24(void) {
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size((void*)0, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size((void*)0, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size((void*)0, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((void*)0, 3);
}

// CIR-LABEL: @test25
// LLVM-LABEL: define {{.*}} void @test25
// OGCG-LABEL: define {{.*}} void @test25
void test25(void) {
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size((void*)0x1000, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size((void*)0x1000, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size((void*)0x1000, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size((void*)0x1000, 3);

  // Skipping (void*)0 + 0x1000 tests - void pointer arithmetic NYI in CIR
}

// CIR-LABEL: @test26
// LLVM-LABEL: define {{.*}} void @test26
// OGCG-LABEL: define {{.*}} void @test26
void test26(void) {
  struct { int v[10]; } t[10];

  // CIR: cir.const #cir.int<316>
  // LLVM: store i32 316
  // OGCG: store i32 316
  gi = __builtin_object_size(&t[1].v[11], 0);
  
  // CIR: cir.const #cir.int<312>
  // LLVM: store i32 312
  // OGCG: store i32 312
  gi = __builtin_object_size(&t[1].v[12], 1);
  
  // CIR: cir.const #cir.int<308>
  // LLVM: store i32 308
  // OGCG: store i32 308
  gi = __builtin_object_size(&t[1].v[13], 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&t[1].v[14], 3);
}

struct Test27IncompleteTy;

// CIR-LABEL: @test27
// LLVM-LABEL: define {{.*}} void @test27
// OGCG-LABEL: define {{.*}} void @test27
void test27(struct Test27IncompleteTy *t) {
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(t, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(t, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(t, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(t, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(&test27, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(&test27, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(&test27, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(&test27, 3);
}

// CIR-LABEL: @test28
// LLVM-LABEL: define {{.*}} void @test28
// OGCG-LABEL: define {{.*}} void @test28
void test28(void) {
  struct { int v[10]; } t[10];

  // CIR: cir.const #cir.int<360>
  // LLVM: store i32 360
  // OGCG: store i32 360
  gi = __builtin_object_size((char*)((short*)(&t[1])), 0);
  
  // CIR: cir.const #cir.int<360>
  // LLVM: store i32 360
  // OGCG: store i32 360
  gi = __builtin_object_size((char*)((short*)(&t[1])), 1);
  
  // CIR: cir.const #cir.int<360>
  // LLVM: store i32 360
  // OGCG: store i32 360
  gi = __builtin_object_size((char*)((short*)(&t[1])), 2);
  
  // CIR: cir.const #cir.int<360>
  // LLVM: store i32 360
  // OGCG: store i32 360
  gi = __builtin_object_size((char*)((short*)(&t[1])), 3);

  // CIR: cir.const #cir.int<356>
  // LLVM: store i32 356
  // OGCG: store i32 356
  gi = __builtin_object_size((char*)((short*)(&t[1].v[1])), 0);
  
  // CIR: cir.const #cir.int<36>
  // LLVM: store i32 36
  // OGCG: store i32 36
  gi = __builtin_object_size((char*)((short*)(&t[1].v[1])), 1);
  
  // CIR: cir.const #cir.int<356>
  // LLVM: store i32 356
  // OGCG: store i32 356
  gi = __builtin_object_size((char*)((short*)(&t[1].v[1])), 2);
  
  // CIR: cir.const #cir.int<36>
  // LLVM: store i32 36
  // OGCG: store i32 36
  gi = __builtin_object_size((char*)((short*)(&t[1].v[1])), 3);
}

struct DynStructVar {
  char fst[16];
  char snd[];
};

struct DynStruct0 {
  char fst[16];
  char snd[0];
};

struct DynStruct1 {
  char fst[16];
  char snd[1];
};

struct StaticStruct {
  char fst[16];
  char snd[2];
};

// CIR-LABEL: @test29
// LLVM-LABEL: define {{.*}} void @test29
// OGCG-LABEL: define {{.*}} void @test29
void test29(struct DynStructVar *dv, struct DynStruct0 *d0,
            struct DynStruct1 *d1, struct StaticStruct *ss) {
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(dv->snd, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(dv->snd, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(dv->snd, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(dv->snd, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(d0->snd, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(d0->snd, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(d0->snd, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(d0->snd, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(d1->snd, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(d1->snd, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(d1->snd, 2);
  
  // CIR: cir.const #cir.int<1>
  // LLVM: store i32 1
  // OGCG: store i32 1
  gi = __builtin_object_size(d1->snd, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(ss->snd, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(ss->snd, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(ss->snd, 2);
  
  // CIR: cir.const #cir.int<2>
  // LLVM: store i32 2
  // OGCG: store i32 2
  gi = __builtin_object_size(ss->snd, 3);
}

// CIR-LABEL: @test30
// LLVM-LABEL: define {{.*}} void @test30
// OGCG-LABEL: define {{.*}} void @test30
void test30(void) {
  struct { struct DynStruct1 fst, snd; } *nested;

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(nested->fst.snd, 0);
  
  // CIR: cir.const #cir.int<1>
  // LLVM: store i32 1
  // OGCG: store i32 1
  gi = __builtin_object_size(nested->fst.snd, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(nested->fst.snd, 2);
  
  // CIR: cir.const #cir.int<1>
  // LLVM: store i32 1
  // OGCG: store i32 1
  gi = __builtin_object_size(nested->fst.snd, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(nested->snd.snd, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(nested->snd.snd, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(nested->snd.snd, 2);
  
  // CIR: cir.const #cir.int<1>
  // LLVM: store i32 1
  // OGCG: store i32 1
  gi = __builtin_object_size(nested->snd.snd, 3);

  union { struct DynStruct1 d1; char c[1]; } *u;
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(u->c, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(u->c, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(u->c, 2);
  
  // CIR: cir.const #cir.int<1>
  // LLVM: store i32 1
  // OGCG: store i32 1
  gi = __builtin_object_size(u->c, 3);

  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(u->d1.snd, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(u->d1.snd, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(u->d1.snd, 2);
  
  // CIR: cir.const #cir.int<1>
  // LLVM: store i32 1
  // OGCG: store i32 1
  gi = __builtin_object_size(u->d1.snd, 3);
}

// CIR-LABEL: @test32
// LLVM-LABEL: define {{.*}} i64 @test32
// OGCG-LABEL: define {{.*}} i64 @test32
static struct DynStructVar D32 = {
  .fst = {},
  .snd = { 0, 1, 2, },
};
unsigned long test32(void) {
  // CIR: cir.const #cir.int<19>
  // LLVM: store i64 19
  // OGCG: ret i64 19
  return __builtin_object_size(&D32, 1);
}

// CIR-LABEL: @test33
// LLVM-LABEL: define {{.*}} i64 @test33
// OGCG-LABEL: define {{.*}} i64 @test33
static struct DynStructVar D33 = {
  .fst = {},
  .snd = {},
};
unsigned long test33(void) {
  // CIR: cir.const #cir.int<16>
  // LLVM: store i64 16
  // OGCG: ret i64 16
  return __builtin_object_size(&D33, 1);
}

// CIR-LABEL: @test34
// LLVM-LABEL: define {{.*}} i64 @test34
// OGCG-LABEL: define {{.*}} i64 @test34
static struct DynStructVar D34 = {
  .fst = {},
};
unsigned long test34(void) {
  // CIR: cir.const #cir.int<16>
  // LLVM: store i64 16
  // OGCG: ret i64 16
  return __builtin_object_size(&D34, 1);
}

// CIR-LABEL: @test35
// LLVM-LABEL: define {{.*}} i64 @test35
// OGCG-LABEL: define {{.*}} i64 @test35
unsigned long test35(void) {
  // CIR: cir.const #cir.int<16>
  // LLVM: store i64 16
  // OGCG: ret i64 16
  return __builtin_object_size(&(struct DynStructVar){}, 1);
}

// CIR-LABEL: @test37
// LLVM-LABEL: define {{.*}} i64 @test37
// OGCG-LABEL: define {{.*}} i64 @test37
struct Z { struct A { int x, y[]; } z; int a; int b[]; };
static struct Z my_z = { .b = {1,2,3} };
unsigned long test37(void) {
  // CIR: cir.const #cir.int<4>
  // LLVM: store i64 4
  // OGCG: ret i64 4
  return __builtin_object_size(&my_z.z, 1);
}

// CIR-LABEL: @PR30346
// LLVM-LABEL: define {{.*}} void @PR30346
// OGCG-LABEL: define {{.*}} void @PR30346
void PR30346(void) {
  struct sa_family_t {};
  struct sockaddr {
    struct sa_family_t sa_family;
    char sa_data[14];
  };

  struct sockaddr *sa;
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(sa->sa_data, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = __builtin_object_size(sa->sa_data, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  // OGCG: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = __builtin_object_size(sa->sa_data, 2);
  
  // CIR: cir.const #cir.int<14>
  // LLVM: store i32 14
  // OGCG: store i32 14
  gi = __builtin_object_size(sa->sa_data, 3);
}

extern char incomplete_char_array[];

// CIR-LABEL: @incomplete_and_function_types
// LLVM-LABEL: define {{.*}} void @incomplete_and_function_types
// OGCG-LABEL: define {{.*}} void @incomplete_and_function_types
void incomplete_and_function_types(void) {
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0
  // OGCG: call i64 @llvm.objectsize.i64.p0
  gi = __builtin_object_size(incomplete_char_array, 0);
  
  // CIR: cir.objsize max nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0
  // OGCG: call i64 @llvm.objectsize.i64.p0
  gi = __builtin_object_size(incomplete_char_array, 1);
  
  // CIR: cir.objsize min nullunknown {{.*}} : !cir.ptr<!void> -> !u64i
  // LLVM: call i64 @llvm.objectsize.i64.p0
  // OGCG: call i64 @llvm.objectsize.i64.p0
  gi = __builtin_object_size(incomplete_char_array, 2);
  
  // CIR: cir.const #cir.int<0>
  // LLVM: store i32 0
  // OGCG: store i32 0
  gi = __builtin_object_size(incomplete_char_array, 3);
}

// CIR-LABEL: @deeply_nested
// LLVM-LABEL: define {{.*}} void @deeply_nested
// OGCG-LABEL: define {{.*}} void @deeply_nested
void deeply_nested(void) {
  struct {
    struct {
      struct {
        struct {
          int e[2];
          char f;
        } d[2];
      } c[2];
    } b[2];
  } *a;

  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&a->b[1].c[1].d[1].e[1], 1);
  
  // CIR: cir.const #cir.int<4>
  // LLVM: store i32 4
  // OGCG: store i32 4
  gi = __builtin_object_size(&a->b[1].c[1].d[1].e[1], 3);
}
