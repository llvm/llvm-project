// RUN: %clang_cc1 -no-enable-noundef-analysis           -triple x86_64-apple-darwin -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -DDYNAMIC -triple x86_64-apple-darwin -emit-llvm %s -o - 2>&1 | FileCheck %s

#ifndef DYNAMIC
#define OBJECT_SIZE_BUILTIN __builtin_object_size
#else
#define OBJECT_SIZE_BUILTIN __builtin_dynamic_object_size
#endif

#define strcpy(dest, src) \
  ((OBJECT_SIZE_BUILTIN(dest, 0) != -1ULL) \
   ? __builtin___strcpy_chk (dest, src, OBJECT_SIZE_BUILTIN(dest, 1)) \
   : __inline_strcpy_chk(dest, src))

static char *__inline_strcpy_chk (char *dest, const char *src) {
  return __builtin___strcpy_chk(dest, src, OBJECT_SIZE_BUILTIN(dest, 1));
}

char gbuf[63];
char *gp;
int gi, gj;

// CHECK-LABEL: define{{.*}} void @test1
void test1(void) {
  // CHECK:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 0, i64 4), ptr @.str, i64 59)
  strcpy(&gbuf[4], "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test2
void test2(void) {
  // CHECK:     = call ptr @__strcpy_chk(ptr @gbuf, ptr @.str, i64 63)
  strcpy(gbuf, "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test3
void test3(void) {
  // CHECK:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 0, i64 100), ptr @.str, i64 0)
  strcpy(&gbuf[100], "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test4
void test4(void) {
  // CHECK:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 0, i64 -1), ptr @.str, i64 0)
  strcpy((char*)(void*)&gbuf[-1], "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test5
void test5(void) {
  // CHECK:     = load ptr, ptr @gp
  // CHECK-NEXT:= call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  strcpy(gp, "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test6
void test6(void) {
  char buf[57];

  // CHECK:       = call ptr @__strcpy_chk(ptr %{{.*}}, ptr @.str, i64 53)
  strcpy(&buf[4], "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test7
void test7(void) {
  int i;
  // Ensure we only evaluate the side-effect once.
  // CHECK:     = add
  // CHECK-NOT: = add
  // CHECK:     = call ptr @__strcpy_chk(ptr @gbuf, ptr @.str, i64 63)
  strcpy((++i, gbuf), "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test8
void test8(void) {
  char *buf[50];
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)
  strcpy(buf[++gi], "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test9
void test9(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)
  strcpy((char *)((++gi) + gj), "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test10
char **p;
void test10(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)
  strcpy(*(++p), "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test11
void test11(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr @gbuf, ptr @.str)
  strcpy(gp = gbuf, "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test12
void test12(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)
  strcpy(++gp, "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test13
void test13(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)
  strcpy(gp++, "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test14
void test14(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)
  strcpy(--gp, "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test15
void test15(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{..*}}, ptr @.str)
  strcpy(gp--, "Hi there");
}

// CHECK-LABEL: define{{.*}} void @test16
void test16(void) {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)
  strcpy(gp += 1, "Hi there");
}

// CHECK-LABEL: @test17
void test17(void) {
  // CHECK: store i32 -1
  gi = OBJECT_SIZE_BUILTIN(gp++, 0);
  // CHECK: store i32 -1
  gi = OBJECT_SIZE_BUILTIN(gp++, 1);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(gp++, 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(gp++, 3);
}

// CHECK-LABEL: @test18
unsigned test18(int cond) {
  int a[4], b[4];
  // CHECK: phi ptr
  // CHECK: call i64 @llvm.objectsize.i64
  return OBJECT_SIZE_BUILTIN(cond ? a : b, 0);
}

// CHECK-LABEL: @test19
void test19(void) {
  struct {
    int a, b;
  } foo;

  // CHECK: store i32 8
  gi = OBJECT_SIZE_BUILTIN(&foo.a, 0);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&foo.a, 1);
  // CHECK: store i32 8
  gi = OBJECT_SIZE_BUILTIN(&foo.a, 2);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&foo.a, 3);

  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&foo.b, 0);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&foo.b, 1);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&foo.b, 2);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&foo.b, 3);
}

// CHECK-LABEL: @test20
void test20(void) {
  struct { int t[10]; } t[10];

  // CHECK: store i32 380
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 0);
  // CHECK: store i32 20
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 1);
  // CHECK: store i32 380
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 2);
  // CHECK: store i32 20
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 3);
}

// CHECK-LABEL: @test21
void test21(void) {
  struct { int t; } t;

  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t + 1, 0);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t + 1, 1);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t + 1, 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t + 1, 3);

  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 0);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 1);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 3);
}

// CHECK-LABEL: @test22
void test22(void) {
  struct { int t[10]; } t[10];

  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[10], 0);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[10], 1);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[10], 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[10], 3);

  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 0);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 1);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 3);

  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 0);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 1);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 3);

  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 0);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 1);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 3);
}

struct Test23Ty { int a; int t[10]; };

// CHECK-LABEL: @test23
void test23(struct Test23Ty *p) {
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(p, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(p, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(p, 2);
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(p, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&p->a, 0);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&p->a, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&p->a, 2);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&p->a, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 2);
  // CHECK: store i32 20
  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 3);
}

// PR24493 -- ICE if OBJECT_SIZE_BUILTIN called with NULL and (Type & 1) != 0
// CHECK-LABEL: @test24
void test24(void) {
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0, 2);
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((void*)0, 3);
}

// CHECK-LABEL: @test25
void test25(void) {
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 2);
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 2);
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 3);
}

// CHECK-LABEL: @test26
void test26(void) {
  struct { int v[10]; } t[10];

  // CHECK: store i32 316
  gi = OBJECT_SIZE_BUILTIN(&t[1].v[11], 0);
  // CHECK: store i32 312
  gi = OBJECT_SIZE_BUILTIN(&t[1].v[12], 1);
  // CHECK: store i32 308
  gi = OBJECT_SIZE_BUILTIN(&t[1].v[13], 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&t[1].v[14], 3);
}

struct Test27IncompleteTy;

// CHECK-LABEL: @test27
void test27(struct Test27IncompleteTy *t) {
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(t, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(t, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(t, 2);
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(t, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&test27, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&test27, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&test27, 2);
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(&test27, 3);
}

// The intent of this test is to ensure that OBJECT_SIZE_BUILTIN treats `&foo`
// and `(T*)&foo` identically, when used as the pointer argument.
// CHECK-LABEL: @test28
void test28(void) {
  struct { int v[10]; } t[10];

#define addCasts(s) ((char*)((short*)(s)))
  // CHECK: store i32 360
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 0);
  // CHECK: store i32 360
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 1);
  // CHECK: store i32 360
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 2);
  // CHECK: store i32 360
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 3);

  // CHECK: store i32 356
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1].v[1]), 0);
  // CHECK: store i32 36
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1].v[1]), 1);
  // CHECK: store i32 356
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1].v[1]), 2);
  // CHECK: store i32 36
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1].v[1]), 3);
#undef addCasts
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

// CHECK-LABEL: @test29
void test29(struct DynStructVar *dv, struct DynStruct0 *d0,
            struct DynStruct1 *d1, struct StaticStruct *ss) {
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(dv->snd, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(dv->snd, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(dv->snd, 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(dv->snd, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(d0->snd, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(d0->snd, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(d0->snd, 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(d0->snd, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(d1->snd, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(d1->snd, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(d1->snd, 2);
  // CHECK: store i32 1
  gi = OBJECT_SIZE_BUILTIN(d1->snd, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(ss->snd, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(ss->snd, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(ss->snd, 2);
  // CHECK: store i32 2
  gi = OBJECT_SIZE_BUILTIN(ss->snd, 3);
}

// CHECK-LABEL: @test30
void test30(void) {
  struct { struct DynStruct1 fst, snd; } *nested;

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 0);
  // CHECK: store i32 1
  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 2);
  // CHECK: store i32 1
  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 2);
  // CHECK: store i32 1
  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 3);

  union { struct DynStruct1 d1; char c[1]; } *u;
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(u->c, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(u->c, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(u->c, 2);
  // CHECK: store i32 1
  gi = OBJECT_SIZE_BUILTIN(u->c, 3);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 2);
  // CHECK: store i32 1
  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 3);
}

// CHECK-LABEL: @test31
void test31(void) {
  // Miscellaneous 'writing off the end' detection tests
  struct DynStructVar *dsv;
  struct DynStruct0 *ds0;
  struct DynStruct1 *ds1;
  struct StaticStruct *ss;

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(ds1[9].snd, 1);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&ss[9].snd[0], 1);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&ds1[9].snd[0], 1);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&ds0[9].snd[0], 1);

  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(&dsv[9].snd[0], 1);
}

// CHECK-LABEL: @test32
static struct DynStructVar D32 = {
  .fst = {},
  .snd = { 0, 1, 2, },
};
unsigned long test32(void) {
  // CHECK: ret i64 19
  return OBJECT_SIZE_BUILTIN(&D32, 1);
}
// CHECK-LABEL: @test33
static struct DynStructVar D33 = {
  .fst = {},
  .snd = {},
};
unsigned long test33(void) {
  // CHECK: ret i64 16
  return OBJECT_SIZE_BUILTIN(&D33, 1);
}
// CHECK-LABEL: @test34
static struct DynStructVar D34 = {
  .fst = {},
};
unsigned long test34(void) {
  // CHECK: ret i64 16
  return OBJECT_SIZE_BUILTIN(&D34, 1);
}
// CHECK-LABEL: @test35
unsigned long test35(void) {
  // CHECK: ret i64 16
  return OBJECT_SIZE_BUILTIN(&(struct DynStructVar){}, 1);
}
extern void *memset (void *s, int c, unsigned long n);
void test36(void) {
  struct DynStructVar D;
  // FORTIFY will check the object size of D. Test this doesn't assert when
  // given a struct with a flexible array member that lacks an initializer.
  memset(&D, 0, sizeof(D));
}
// CHECK-LABEL: @test37
struct Z { struct A { int x, y[]; } z; int a; int b[]; };
static struct Z my_z = { .b = {1,2,3} };
unsigned long test37 (void) {
  // CHECK: ret i64 4
  return OBJECT_SIZE_BUILTIN(&my_z.z, 1);
}

// CHECK-LABEL: @PR30346
void PR30346(void) {
  struct sa_family_t {};
  struct sockaddr {
    struct sa_family_t sa_family;
    char sa_data[14];
  };

  struct sockaddr *sa;
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1
  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 2);
  // CHECK: store i32 14
  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 3);
}

extern char incomplete_char_array[];
// CHECK-LABEL: @incomplete_and_function_types
int incomplete_and_function_types(void) {
  // CHECK: call i64 @llvm.objectsize.i64.p0
  gi = OBJECT_SIZE_BUILTIN(incomplete_char_array, 0);
  // CHECK: call i64 @llvm.objectsize.i64.p0
  gi = OBJECT_SIZE_BUILTIN(incomplete_char_array, 1);
  // CHECK: call i64 @llvm.objectsize.i64.p0
  gi = OBJECT_SIZE_BUILTIN(incomplete_char_array, 2);
  // CHECK: store i32 0
  gi = OBJECT_SIZE_BUILTIN(incomplete_char_array, 3);
}

// Flips between the pointer and lvalue evaluator a lot.
void deeply_nested(void) {
  struct {
    struct {
      struct {
        struct {
          int e[2];
          char f; // Inhibit our writing-off-the-end check
        } d[2];
      } c[2];
    } b[2];
  } *a;

  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&a->b[1].c[1].d[1].e[1], 1);
  // CHECK: store i32 4
  gi = OBJECT_SIZE_BUILTIN(&a->b[1].c[1].d[1].e[1], 3);
}
