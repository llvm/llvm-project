// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - 2>&1 | FileCheck %s

#define strcpy(dest, src) \
  ((__builtin_object_size(dest, 0) != -1ULL) \
   ? __builtin___strcpy_chk (dest, src, __builtin_object_size(dest, 1)) \
   : __inline_strcpy_chk(dest, src))

static char *__inline_strcpy_chk (char *dest, const char *src) {
  return __builtin___strcpy_chk(dest, src, __builtin_object_size(dest, 1));
}

char gbuf[63];
char *gp;
int gi, gj;

// CHECK: define void @test1
void test1() {
  // CHECK:     = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i64 4), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 59)
  strcpy(&gbuf[4], "Hi there");
}

// CHECK: define void @test2
void test2() {
  // CHECK:     = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 63)
  strcpy(gbuf, "Hi there");
}

// CHECK: define void @test3
void test3() {
  // CHECK:     = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i64 1, i64 37), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 0)
  strcpy(&gbuf[100], "Hi there");
}

// CHECK: define void @test4
void test4() {
  // CHECK:     = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i64 -1), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 0)
  strcpy((char*)(void*)&gbuf[-1], "Hi there");
}

// CHECK: define void @test5
void test5() {
  // CHECK:     = load i8** @gp
  // CHECK-NEXT:= call i64 @llvm.objectsize.i64(i8* %{{.*}}, i1 false)
  strcpy(gp, "Hi there");
}

// CHECK: define void @test6
void test6() {
  char buf[57];

  // CHECK:       = call i8* @__strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 53)
  strcpy(&buf[4], "Hi there");
}

// CHECK: define void @test7
void test7() {
  int i;
  // Ensure we only evaluate the side-effect once.
  // CHECK:     = add
  // CHECK-NOT: = add
  // CHECK:     = call i8* @__strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), i64 63)
  strcpy((++i, gbuf), "Hi there");
}

// CHECK: define void @test8
void test8() {
  char *buf[50];
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(buf[++gi], "Hi there");
}

// CHECK: define void @test9
void test9() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy((char *)((++gi) + gj), "Hi there");
}

// CHECK: define void @test10
char **p;
void test10() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(*(++p), "Hi there");
}

// CHECK: define void @test11
void test11() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* getelementptr inbounds ([63 x i8]* @gbuf, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp = gbuf, "Hi there");
}

// CHECK: define void @test12
void test12() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(++gp, "Hi there");
}

// CHECK: define void @test13
void test13() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp++, "Hi there");
}

// CHECK: define void @test14
void test14() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(--gp, "Hi there");
}

// CHECK: define void @test15
void test15() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{..*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp--, "Hi there");
}

// CHECK: define void @test16
void test16() {
  // CHECK-NOT:   __strcpy_chk
  // CHECK:       = call i8* @__inline_strcpy_chk(i8* %{{.*}}, i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0))
  strcpy(gp += 1, "Hi there");
}

// CHECK: @test17
void test17() {
  // CHECK: store i32 -1
  gi = __builtin_object_size(gp++, 0);
  // CHECK: store i32 -1
  gi = __builtin_object_size(gp++, 1);
  // CHECK: store i32 0
  gi = __builtin_object_size(gp++, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(gp++, 3);
}

// CHECK: @test18
unsigned test18(int cond) {
  int a[4], b[4];
  // CHECK: phi i32*
  // CHECK: call i64 @llvm.objectsize.i64
  return __builtin_object_size(cond ? a : b, 0);
}
