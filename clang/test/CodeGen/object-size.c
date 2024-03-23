// RUN: %clang_cc1 -no-enable-noundef-analysis           -triple x86_64-apple-darwin -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=STATIC  %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -DDYNAMIC -triple x86_64-apple-darwin -emit-llvm %s -o - 2>&1 | FileCheck --check-prefix=DYNAMIC %s

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

void test1(void) {
  // STATIC-LABEL: define{{.*}} void @test1
  // STATIC:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 0, i64 4), ptr @.str, i64 59)

  // DYNAMIC-LABEL: define{{.*}} void @test1
  // DYNAMIC:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 0, i64 4), ptr @.str, i64 59)

  strcpy(&gbuf[4], "Hi there");
}

void test2(void) {
  // STATIC-LABEL: define{{.*}} void @test2
  // STATIC:     = call ptr @__strcpy_chk(ptr @gbuf, ptr @.str, i64 63)

  // DYNAMIC-LABEL: define{{.*}} void @test2
  // DYNAMIC:     = call ptr @__strcpy_chk(ptr @gbuf, ptr @.str, i64 63)

  strcpy(gbuf, "Hi there");
}

void test3(void) {
  // STATIC-LABEL: define{{.*}} void @test3
  // STATIC:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 1, i64 37), ptr @.str, i64 0)

  // DYNAMIC-LABEL: define{{.*}} void @test3
  // DYNAMIC:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 1, i64 37), ptr @.str, i64 0)

  strcpy(&gbuf[100], "Hi there");
}

void test4(void) {
  // STATIC-LABEL: define{{.*}} void @test4
  // STATIC:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 0, i64 -1), ptr @.str, i64 0)

  // DYNAMIC-LABEL: define{{.*}} void @test4
  // DYNAMIC:     = call ptr @__strcpy_chk(ptr getelementptr inbounds ([63 x i8], ptr @gbuf, i64 0, i64 -1), ptr @.str, i64 0)

  strcpy((char*)(void*)&gbuf[-1], "Hi there");
}

void test5(void) {
  // STATIC-LABEL: define{{.*}} void @test5
  // STATIC:     = load ptr, ptr @gp
  // STATIC-NEXT:= call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)

  // DYNAMIC-LABEL: define{{.*}} void @test5
  // DYNAMIC:     = load ptr, ptr @gp
  // DYNAMIC-NEXT:= call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)

  strcpy(gp, "Hi there");
}

void test6(void) {
  char buf[57];

  // STATIC-LABEL: define{{.*}} void @test6
  // STATIC:       = call ptr @__strcpy_chk(ptr %{{.*}}, ptr @.str, i64 53)

  // DYNAMIC-LABEL: define{{.*}} void @test6
  // DYNAMIC:       = call ptr @__strcpy_chk(ptr %{{.*}}, ptr @.str, i64 53)

  strcpy(&buf[4], "Hi there");
}

void test7(void) {
  // Ensure we only evaluate the side-effect once.
  int i;

  // STATIC-LABEL: define{{.*}} void @test7
  // STATIC:     = add
  // STATIC-NOT: = add
  // STATIC:     = call ptr @__strcpy_chk(ptr @gbuf, ptr @.str, i64 63)

  // DYNAMIC-LABEL: define{{.*}} void @test7
  // DYNAMIC:     = add
  // DYNAMIC-NOT: = add
  // DYNAMIC:     = call ptr @__strcpy_chk(ptr @gbuf, ptr @.str, i64 63)

  strcpy((++i, gbuf), "Hi there");
}

void test8(void) {
  char *buf[50];
  // STATIC-LABEL: define{{.*}} void @test8
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test8
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  strcpy(buf[++gi], "Hi there");
}

void test9(void) {
  // STATIC-LABEL: define{{.*}} void @test9
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test9
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  strcpy((char *)((++gi) + gj), "Hi there");
}

char **p;
void test10(void) {
  // STATIC-LABEL: define{{.*}} void @test10
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test10
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  strcpy(*(++p), "Hi there");
}

void test11(void) {
  // STATIC-LABEL: define{{.*}} void @test11
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr @gbuf, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test11
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr @gbuf, ptr @.str)

  strcpy(gp = gbuf, "Hi there");
}

void test12(void) {
  // STATIC-LABEL: define{{.*}} void @test12
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test12
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  strcpy(++gp, "Hi there");
}

void test13(void) {
  // STATIC-LABEL: define{{.*}} void @test13
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test13
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  strcpy(gp++, "Hi there");
}

void test14(void) {
  // STATIC-LABEL: define{{.*}} void @test14
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test14
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  strcpy(--gp, "Hi there");
}

void test15(void) {
  // STATIC-LABEL: define{{.*}} void @test15
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{..*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test15
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{..*}}, ptr @.str)

  strcpy(gp--, "Hi there");
}

void test16(void) {
  // STATIC-LABEL: define{{.*}} void @test16
  // STATIC-NOT:   __strcpy_chk
  // STATIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  // DYNAMIC-LABEL: define{{.*}} void @test16
  // DYNAMIC-NOT:   __strcpy_chk
  // DYNAMIC:       = call ptr @__inline_strcpy_chk(ptr %{{.*}}, ptr @.str)

  strcpy(gp += 1, "Hi there");
}

void test17(void) {
  // STATIC-LABEL: @test17
  // STATIC: store i32 -1
  // STATIC: store i32 -1
  // STATIC: store i32 0
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test17
  // DYNAMIC: store i32 -1
  // DYNAMIC: store i32 -1
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(gp++, 0);
  gi = OBJECT_SIZE_BUILTIN(gp++, 1);
  gi = OBJECT_SIZE_BUILTIN(gp++, 2);
  gi = OBJECT_SIZE_BUILTIN(gp++, 3);
}

unsigned test18(int cond) {
  int a[4], b[4];
  // STATIC-LABEL: @test18
  // STATIC: phi ptr
  // STATIC: call i64 @llvm.objectsize.i64

  // DYNAMIC-LABEL: @test18
  // DYNAMIC: phi ptr
  // DYNAMIC: call i64 @llvm.objectsize.i64

  return OBJECT_SIZE_BUILTIN(cond ? a : b, 0);
}

void test19(void) {
  struct {
    int a, b;
  } foo;

  // STATIC-LABEL: @test19
  // STATIC: store i32 8
  // STATIC: store i32 4
  // STATIC: store i32 8
  // STATIC: store i32 4

  // DYNAMIC-LABEL: @test19
  // DYNAMIC: store i32 8
  // DYNAMIC: store i32 4
  // DYNAMIC: store i32 8
  // DYNAMIC: store i32 4

  gi = OBJECT_SIZE_BUILTIN(&foo.a, 0);
  gi = OBJECT_SIZE_BUILTIN(&foo.a, 1);
  gi = OBJECT_SIZE_BUILTIN(&foo.a, 2);
  gi = OBJECT_SIZE_BUILTIN(&foo.a, 3);

  // STATIC: store i32 4
  // STATIC: store i32 4
  // STATIC: store i32 4
  // STATIC: store i32 4

  // DYNAMIC: store i32 4
  // DYNAMIC: store i32 4
  // DYNAMIC: store i32 4
  // DYNAMIC: store i32 4

  gi = OBJECT_SIZE_BUILTIN(&foo.b, 0);
  gi = OBJECT_SIZE_BUILTIN(&foo.b, 1);
  gi = OBJECT_SIZE_BUILTIN(&foo.b, 2);
  gi = OBJECT_SIZE_BUILTIN(&foo.b, 3);
}

void test20(void) {
  struct { int t[10]; } t[10];

  // STATIC-LABEL: @test20
  // STATIC: store i32 380
  // STATIC: store i32 20
  // STATIC: store i32 380
  // STATIC: store i32 20

  // DYNAMIC-LABEL: @test20
  // DYNAMIC: store i32 380
  // DYNAMIC: store i32 20
  // DYNAMIC: store i32 380

  // DYNAMIC: store i32 20
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 0);
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 1);
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 2);
  gi = OBJECT_SIZE_BUILTIN(&t[0].t[5], 3);
}

void test21(void) {
  struct { int t; } t;

  // STATIC-LABEL: @test21
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test21
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(&t + 1, 0);
  gi = OBJECT_SIZE_BUILTIN(&t + 1, 1);
  gi = OBJECT_SIZE_BUILTIN(&t + 1, 2);
  gi = OBJECT_SIZE_BUILTIN(&t + 1, 3);

  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0

  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 0);
  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 1);
  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 2);
  gi = OBJECT_SIZE_BUILTIN(&t.t + 1, 3);
}

void test22(void) {
  struct { int t[10]; } t[10];

  // STATIC-LABEL: @test22
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test22
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(&t[10], 0);
  gi = OBJECT_SIZE_BUILTIN(&t[10], 1);
  gi = OBJECT_SIZE_BUILTIN(&t[10], 2);
  gi = OBJECT_SIZE_BUILTIN(&t[10], 3);

  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0

  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 0);
  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 1);
  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 2);
  gi = OBJECT_SIZE_BUILTIN(&t[9].t[10], 3);

  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0

  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 0);
  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 1);
  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 2);
  gi = OBJECT_SIZE_BUILTIN((char*)&t[0] + sizeof(t), 3);

  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0
  // STATIC: store i32 0

  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 0);
  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 1);
  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 2);
  gi = OBJECT_SIZE_BUILTIN((char*)&t[9].t[0] + 10*sizeof(t[0].t), 3);
}

struct Test23Ty { int a; int t[10]; };

void test23(struct Test23Ty *p) {
  // STATIC-LABEL: @test23
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test23
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(p, 0);
  gi = OBJECT_SIZE_BUILTIN(p, 1);
  gi = OBJECT_SIZE_BUILTIN(p, 2);
  gi = OBJECT_SIZE_BUILTIN(p, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 4
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 4

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 4
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 4

  gi = OBJECT_SIZE_BUILTIN(&p->a, 0);
  gi = OBJECT_SIZE_BUILTIN(&p->a, 1);
  gi = OBJECT_SIZE_BUILTIN(&p->a, 2);
  gi = OBJECT_SIZE_BUILTIN(&p->a, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 20

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 40)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 20

  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 0);
  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 1);
  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 2);
  gi = OBJECT_SIZE_BUILTIN(&p->t[5], 3);
}

// PR24493 -- ICE if OBJECT_SIZE_BUILTIN called with NULL and (Type & 1) != 0
void test24(void) {
  // STATIC-LABEL: @test24
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test24
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN((void*)0, 0);
  gi = OBJECT_SIZE_BUILTIN((void*)0, 1);
  gi = OBJECT_SIZE_BUILTIN((void*)0, 2);
  gi = OBJECT_SIZE_BUILTIN((void*)0, 3);
}

void test25(void) {
  // STATIC-LABEL: @test25
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test25
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 0);
  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 1);
  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 2);
  gi = OBJECT_SIZE_BUILTIN((void*)0x1000, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // STATIC: store i32 0

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // Note: Currently fixed at zero because LLVM can't handle type=3 correctly.
  // Hopefully will be lowered properly in the future.
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 0);
  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 1);
  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 2);
  gi = OBJECT_SIZE_BUILTIN((void*)0 + 0x1000, 3);
}

void test26(void) {
  struct { int v[10]; } t[10];

  // STATIC-LABEL: @test26
  // STATIC: store i32 316
  // STATIC: store i32 312
  // STATIC: store i32 308
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test26
  // DYNAMIC: store i32 316
  // DYNAMIC: store i32 312
  // DYNAMIC: store i32 308
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(&t[1].v[11], 0);
  gi = OBJECT_SIZE_BUILTIN(&t[1].v[12], 1);
  gi = OBJECT_SIZE_BUILTIN(&t[1].v[13], 2);
  gi = OBJECT_SIZE_BUILTIN(&t[1].v[14], 3);
}

struct Test27IncompleteTy;

void test27(struct Test27IncompleteTy *t) {
  // STATIC-LABEL: @test27
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test27
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(t, 0);
  gi = OBJECT_SIZE_BUILTIN(t, 1);
  gi = OBJECT_SIZE_BUILTIN(t, 2);
  gi = OBJECT_SIZE_BUILTIN(t, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // STATIC: store i32 0

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // Note: this is currently fixed at 0 because LLVM doesn't have sufficient
  // data to correctly handle type=3
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(&test27, 0);
  gi = OBJECT_SIZE_BUILTIN(&test27, 1);
  gi = OBJECT_SIZE_BUILTIN(&test27, 2);
  gi = OBJECT_SIZE_BUILTIN(&test27, 3);
}

// The intent of this test is to ensure that OBJECT_SIZE_BUILTIN treats `&foo`
// and `(T*)&foo` identically, when used as the pointer argument.
void test28(void) {
  struct { int v[10]; } t[10];

#define addCasts(s) ((char*)((short*)(s)))
  // STATIC-LABEL: @test28
  // STATIC: store i32 360
  // STATIC: store i32 360
  // STATIC: store i32 360
  // STATIC: store i32 360

  // DYNAMIC-LABEL: @test28
  // DYNAMIC: store i32 360
  // DYNAMIC: store i32 360
  // DYNAMIC: store i32 360
  // DYNAMIC: store i32 360

  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 0);
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 1);
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 2);
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1]), 3);

  // STATIC: store i32 356
  // STATIC: store i32 36
  // STATIC: store i32 356
  // STATIC: store i32 36

  // DYNAMIC: store i32 356
  // DYNAMIC: store i32 36
  // DYNAMIC: store i32 356
  // DYNAMIC: store i32 36

  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1].v[1]), 0);
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1].v[1]), 1);
  gi = OBJECT_SIZE_BUILTIN(addCasts(&t[1].v[1]), 2);
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

void test29(struct DynStructVar *dv, struct DynStruct0 *d0,
            struct DynStruct1 *d1, struct StaticStruct *ss) {
  // STATIC-LABEL: @test29
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @test29
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(dv->snd, 0);
  gi = OBJECT_SIZE_BUILTIN(dv->snd, 1);
  gi = OBJECT_SIZE_BUILTIN(dv->snd, 2);
  gi = OBJECT_SIZE_BUILTIN(dv->snd, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 0

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(d0->snd, 0);
  gi = OBJECT_SIZE_BUILTIN(d0->snd, 1);
  gi = OBJECT_SIZE_BUILTIN(d0->snd, 2);
  gi = OBJECT_SIZE_BUILTIN(d0->snd, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 1

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 1)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 1

  gi = OBJECT_SIZE_BUILTIN(d1->snd, 0);
  gi = OBJECT_SIZE_BUILTIN(d1->snd, 1);
  gi = OBJECT_SIZE_BUILTIN(d1->snd, 2);
  gi = OBJECT_SIZE_BUILTIN(d1->snd, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 2

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 2)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 2

  gi = OBJECT_SIZE_BUILTIN(ss->snd, 0);
  gi = OBJECT_SIZE_BUILTIN(ss->snd, 1);
  gi = OBJECT_SIZE_BUILTIN(ss->snd, 2);
  gi = OBJECT_SIZE_BUILTIN(ss->snd, 3);
}

void test30(void) {
  struct { struct DynStruct1 fst, snd; } *nested;

  // STATIC-LABEL: @test30
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 1
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 1

  // DYNAMIC-LABEL: @test30
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 1
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 1

  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 0);
  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 1);
  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 2);
  gi = OBJECT_SIZE_BUILTIN(nested->fst.snd, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 1

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 1)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 1

  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 0);
  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 1);
  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 2);
  gi = OBJECT_SIZE_BUILTIN(nested->snd.snd, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 1

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 1)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 1

  union { struct DynStruct1 d1; char c[1]; } *u;

  gi = OBJECT_SIZE_BUILTIN(u->c, 0);
  gi = OBJECT_SIZE_BUILTIN(u->c, 1);
  gi = OBJECT_SIZE_BUILTIN(u->c, 2);
  gi = OBJECT_SIZE_BUILTIN(u->c, 3);

  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 1

  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 1)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 1

  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 0);
  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 1);
  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 2);
  gi = OBJECT_SIZE_BUILTIN(u->d1.snd, 3);
}

void test31(void) {
  // Miscellaneous 'writing off the end' detection tests
  struct DynStructVar *dsv;
  struct DynStruct0 *ds0;
  struct DynStruct1 *ds1;
  struct StaticStruct *ss;

  // STATIC-LABEL: @test31
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)

  // DYNAMIC-LABEL: @test31
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 1)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 2)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 1)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 0)

  gi = OBJECT_SIZE_BUILTIN(ds1[9].snd, 1);
  gi = OBJECT_SIZE_BUILTIN(&ss[9].snd[0], 1);
  gi = OBJECT_SIZE_BUILTIN(&ds1[9].snd[0], 1);
  gi = OBJECT_SIZE_BUILTIN(&ds0[9].snd[0], 1);
  gi = OBJECT_SIZE_BUILTIN(&dsv[9].snd[0], 1);
}

static struct DynStructVar D32 = {
  .fst = {},
  .snd = { 0, 1, 2, },
};
unsigned long test32(void) {
  // STATIC-LABEL: @test32
  // STATIC: ret i64 19

  // DYNAMIC-LABEL: @test32
  // DYNAMIC: ret i64 19

  return OBJECT_SIZE_BUILTIN(&D32, 1);
}
static struct DynStructVar D33 = {
  .fst = {},
  .snd = {},
};
unsigned long test33(void) {
  // STATIC-LABEL: @test33
  // STATIC: ret i64 16

  // DYNAMIC-LABEL: @test33
  // DYNAMIC: ret i64 16

  return OBJECT_SIZE_BUILTIN(&D33, 1);
}

static struct DynStructVar D34 = {
  .fst = {},
};
unsigned long test34(void) {
  // STATIC-LABEL: @test34
  // STATIC: ret i64 16

  // DYNAMIC-LABEL: @test34
  // DYNAMIC: ret i64 16

  return OBJECT_SIZE_BUILTIN(&D34, 1);
}
unsigned long test35(void) {
  // STATIC-LABEL: @test35
  // STATIC: ret i64 16

  // DYNAMIC-LABEL: @test35
  // DYNAMIC: ret i64 16

  return OBJECT_SIZE_BUILTIN(&(struct DynStructVar){}, 1);
}
extern void *memset (void *s, int c, unsigned long n);
void test36(void) {
  struct DynStructVar D;
  // FORTIFY will check the object size of D. Test this doesn't assert when
  // given a struct with a flexible array member that lacks an initializer.
  memset(&D, 0, sizeof(D));
}
struct Z { struct A { int x, y[]; } z; int a; int b[]; };
static struct Z my_z = { .b = {1,2,3} };
unsigned long test37 (void) {
  // STATIC-LABEL: @test37
  // STATIC: ret i64 4

  // DYNAMIC-LABEL: @test37
  // DYNAMIC: ret i64 4

  return OBJECT_SIZE_BUILTIN(&my_z.z, 1);
}

void PR30346(void) {
  struct sa_family_t {};
  struct sockaddr {
    struct sa_family_t sa_family;
    char sa_data[14];
  };

  struct sockaddr *sa;

  // STATIC-LABEL: @PR30346
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 true, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 false, i1 false, i64 0)
  // STATIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 false, i1 true, i64 0)
  // STATIC: store i32 14

  // DYNAMIC-LABEL: @PR30346
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 false, i1 true, i1 true, i1 false, i64 14)
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0(ptr %{{.*}}, i1 true, i1 true, i1 true, i1 true, i64 0)
  // DYNAMIC: store i32 14

  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 0);
  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 1);
  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 2);
  gi = OBJECT_SIZE_BUILTIN(sa->sa_data, 3);
}

extern char incomplete_char_array[];
int incomplete_and_function_types(void) {
  // STATIC-LABEL: @incomplete_and_function_types
  // STATIC: call i64 @llvm.objectsize.i64.p0
  // STATIC: call i64 @llvm.objectsize.i64.p0
  // STATIC: call i64 @llvm.objectsize.i64.p0
  // STATIC: store i32 0

  // DYNAMIC-LABEL: @incomplete_and_function_types
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0
  // DYNAMIC: call i64 @llvm.objectsize.i64.p0
  // DYNAMIC: store i32 0

  gi = OBJECT_SIZE_BUILTIN(incomplete_char_array, 0);
  gi = OBJECT_SIZE_BUILTIN(incomplete_char_array, 1);
  gi = OBJECT_SIZE_BUILTIN(incomplete_char_array, 2);
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

  // STATIC-LABEL: @deeply_nested
  // STATIC: store i32 4
  // STATIC: store i32 4

  // DYNAMIC-LABEL: @deeply_nested
  // DYNAMIC: store i32 4
  // DYNAMIC: store i32 4

  gi = OBJECT_SIZE_BUILTIN(&a->b[1].c[1].d[1].e[1], 1);
  gi = OBJECT_SIZE_BUILTIN(&a->b[1].c[1].d[1].e[1], 3);
}
