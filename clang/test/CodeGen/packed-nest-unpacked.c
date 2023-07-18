// RUN: %clang_cc1 %s -triple x86_64-apple-macosx10.7.2 -emit-llvm -o - | FileCheck %s

struct X { int x[6]; };
struct Y { char x[13]; struct X y; } __attribute((packed));
struct Y g;
void f(struct X);
struct X foo(void);

// <rdar://problem/10463337>
struct X test1(void) {
  // CHECK: @test1
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr align 1 getelementptr inbounds (%struct.Y, ptr @g, i32 0, i32 1), i64 24, i1 false)
  return g.y;
}
struct X test2(void) {
  // CHECK: @test2
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr align 1 getelementptr inbounds (%struct.Y, ptr @g, i32 0, i32 1), i64 24, i1 false)
  struct X a = g.y;
  return a;
}

void test3(struct X a) {
  // CHECK: @test3
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 getelementptr inbounds (%struct.Y, ptr @g, i32 0, i32 1), ptr {{.*}}, i64 24, i1 false)
  g.y = a;
}

// <rdar://problem/10530444>
void test4(void) {
  // CHECK: @test4
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr align 1 getelementptr inbounds (%struct.Y, ptr @g, i32 0, i32 1), i64 24, i1 false)
  f(g.y);
}

// PR12395
int test5(void) {
  // CHECK: @test5
  // CHECK: load i32, ptr getelementptr inbounds (%struct.Y, ptr @g, i32 0, i32 1), align 1
  return g.y.x[0];
}

// <rdar://problem/11220251>
void test6(void) {
  // CHECK: @test6
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 getelementptr inbounds (%struct.Y, ptr @g, i32 0, i32 1), ptr align 4 %{{.*}}, i64 24, i1 false)
  g.y = foo();
}


struct XBitfield {
  unsigned b1 : 10;
  unsigned b2 : 12;
  unsigned b3 : 10;
};
struct YBitfield {
  char x;
  struct XBitfield y;
} __attribute((packed));
struct YBitfield gbitfield;

unsigned test7(void) {
  // CHECK: @test7
  // CHECK: load i32, ptr getelementptr inbounds (%struct.YBitfield, ptr @gbitfield, i32 0, i32 1), align 1
  return gbitfield.y.b2;
}

void test8(unsigned x) {
  // CHECK: @test8
  // CHECK: load i32, ptr getelementptr inbounds (%struct.YBitfield, ptr @gbitfield, i32 0, i32 1), align 1
  // CHECK: store i32 {{.*}}, ptr getelementptr inbounds (%struct.YBitfield, ptr @gbitfield, i32 0, i32 1), align 1
  gbitfield.y.b2 = x;
}

struct TBitfield
{
  long a;
  char b;
  unsigned c:15;
};
struct TBitfield tbitfield;

unsigned test9(void) {
  // CHECK: @test9
  // CHECK: load i16, ptr getelementptr inbounds (%struct.TBitfield, ptr @tbitfield, i32 0, i32 2), align 1
  return tbitfield.c;
}

void test10(unsigned x) {
  // CHECK: @test10
  // CHECK: load i16, ptr getelementptr inbounds (%struct.TBitfield, ptr @tbitfield, i32 0, i32 2), align 1
  // CHECK: store i16 {{.*}}, ptr getelementptr inbounds (%struct.TBitfield, ptr @tbitfield, i32 0, i32 2), align 1
  tbitfield.c = x;
}

