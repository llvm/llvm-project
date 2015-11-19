// RUN: %clang_cc1 -triple x86_64 -emit-llvm -o - %s | opt -S -strip -o %t
// RUN: FileCheck --check-prefix=CHECK-GLOBAL < %t %s
// RUN: FileCheck --check-prefix=CHECK-FUNCTIONS < %t %s

struct s0 {
  int x;
  int y __attribute__((packed));
};

// CHECK-GLOBAL: @s0_align_x = global i32 4

// CHECK-GLOBAL: @s0_align_y = global i32 1

// CHECK-GLOBAL: @s0_align = global i32 4
int s0_align_x = __alignof(((struct s0*)0)->x);
int s0_align_y = __alignof(((struct s0*)0)->y);
int s0_align   = __alignof(struct s0);

// CHECK-FUNCTIONS-LABEL: define i32 @s0_load_x
// CHECK-FUNCTIONS: [[s0_load_x:%.*]] = load i32, i32* {{.*}}, align 4
// CHECK-FUNCTIONS: ret i32 [[s0_load_x]]
int s0_load_x(struct s0 *a) { return a->x; }
// FIXME: This seems like it should be align 1. This is actually something which
// has changed in llvm-gcc recently, previously both x and y would be loaded
// with align 1 (in 2363.1 at least).
//
// CHECK-FUNCTIONS-LABEL: define i32 @s0_load_y
// CHECK-FUNCTIONS: [[s0_load_y:%.*]] = load i32, i32* {{.*}}, align 4
// CHECK-FUNCTIONS: ret i32 [[s0_load_y]]
int s0_load_y(struct s0 *a) { return a->y; }
// CHECK-FUNCTIONS-LABEL: define void @s0_copy
// CHECK-FUNCTIONS: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 {{.*}}, i8* align 4 {{.*}}, i64 8, i1 false)
void s0_copy(struct s0 *a, struct s0 *b) { *b = *a; }

//

struct s1 {
  int x;
  int y;
} __attribute__((packed));

// CHECK-GLOBAL: @s1_align_x = global i32 1
// CHECK-GLOBAL: @s1_align_y = global i32 1
// CHECK-GLOBAL: @s1_align = global i32 1
int s1_align_x = __alignof(((struct s1*)0)->x);
int s1_align_y = __alignof(((struct s1*)0)->y);
int s1_align   = __alignof(struct s1);

// CHECK-FUNCTIONS-LABEL: define i32 @s1_load_x
// CHECK-FUNCTIONS: [[s1_load_x:%.*]] = load i32, i32* {{.*}}, align 1
// CHECK-FUNCTIONS: ret i32 [[s1_load_x]]
int s1_load_x(struct s1 *a) { return a->x; }
// CHECK-FUNCTIONS-LABEL: define i32 @s1_load_y
// CHECK-FUNCTIONS: [[s1_load_y:%.*]] = load i32, i32* {{.*}}, align 1
// CHECK-FUNCTIONS: ret i32 [[s1_load_y]]
int s1_load_y(struct s1 *a) { return a->y; }
// CHECK-FUNCTIONS-LABEL: define void @s1_copy
// CHECK-FUNCTIONS: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 {{.*}}, i8* align 1 {{.*}}, i64 8, i1 false)
void s1_copy(struct s1 *a, struct s1 *b) { *b = *a; }

//

#pragma pack(push,2)
struct s2 {
  int x;
  int y;
};
#pragma pack(pop)

// CHECK-GLOBAL: @s2_align_x = global i32 2
// CHECK-GLOBAL: @s2_align_y = global i32 2
// CHECK-GLOBAL: @s2_align = global i32 2
int s2_align_x = __alignof(((struct s2*)0)->x);
int s2_align_y = __alignof(((struct s2*)0)->y);
int s2_align   = __alignof(struct s2);

// CHECK-FUNCTIONS-LABEL: define i32 @s2_load_x
// CHECK-FUNCTIONS: [[s2_load_y:%.*]] = load i32, i32* {{.*}}, align 2
// CHECK-FUNCTIONS: ret i32 [[s2_load_y]]
int s2_load_x(struct s2 *a) { return a->x; }
// CHECK-FUNCTIONS-LABEL: define i32 @s2_load_y
// CHECK-FUNCTIONS: [[s2_load_y:%.*]] = load i32, i32* {{.*}}, align 2
// CHECK-FUNCTIONS: ret i32 [[s2_load_y]]
int s2_load_y(struct s2 *a) { return a->y; }
// CHECK-FUNCTIONS-LABEL: define void @s2_copy
// CHECK-FUNCTIONS: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 {{.*}}, i8* align 2 {{.*}}, i64 8, i1 false)
void s2_copy(struct s2 *a, struct s2 *b) { *b = *a; }

struct __attribute__((packed, aligned)) s3 {
  short aShort;
  int anInt;
};
// CHECK-GLOBAL: @s3_1 = global i32 1
int s3_1 = __alignof(((struct s3*) 0)->anInt);
// CHECK-FUNCTIONS-LABEL: define i32 @test3(
int test3(struct s3 *ptr) {
  // CHECK-FUNCTIONS:      [[PTR:%.*]] = getelementptr inbounds {{%.*}}, {{%.*}}* {{%.*}}, i32 0, i32 1
  // CHECK-FUNCTIONS-NEXT: load i32, i32* [[PTR]], align 2
  return ptr->anInt;
}
