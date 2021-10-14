// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include "Inputs/stdio.h"
#include <stdint.h>

// CHECK: @__const.unit1.a = private unnamed_addr constant %struct.U1A { i16 12 }, align 2
// CHECK-NEXT: [[STRUCT_STR_U1:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U1A {\0A\00"
// CHECK-NEXT: [[FIELD_U1:@[0-9]+]] = private unnamed_addr constant [11 x i8] c"short a : \00"
// CHECK-NEXT: [[FORMAT_U1:@[0-9]+]] = private unnamed_addr constant [5 x i8] c"%hd\0A\00"
// CHECK-NEXT: [[END_STRUCT_U1:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit2.a = private unnamed_addr constant %struct.U2A { i16 12 }, align 2
// CHECK-NEXT: [[STRUCT_STR_U2:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U2A {\0A\00"
// CHECK-NEXT: [[FIELD_U2:@[0-9]+]] = private unnamed_addr constant [20 x i8] c"unsigned short a : \00"
// CHECK-NEXT: [[FORMAT_U2:@[0-9]+]] = private unnamed_addr constant [5 x i8] c"%hu\0A\00"
// CHECK-NEXT: [[END_STRUCT_U2:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit3.a = private unnamed_addr constant %struct.U3A { i32 12 }, align 4
// CHECK-NEXT: [[STRUCT_STR_U3:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U3A {\0A\00"
// CHECK-NEXT: [[FIELD_U3:@[0-9]+]] = private unnamed_addr constant [9 x i8] c"int a : \00"
// CHECK-NEXT: [[FORMAT_U3:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%d\0A\00"
// CHECK-NEXT: [[END_STRUCT_U3:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit4.a = private unnamed_addr constant %struct.U4A { i32 12 }, align 4
// CHECK-NEXT: [[STRUCT_STR_U4:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U4A {\0A\00"
// CHECK-NEXT: [[FIELD_U4:@[0-9]+]] = private unnamed_addr constant [18 x i8] c"unsigned int a : \00"
// CHECK-NEXT: [[FORMAT_U4:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%u\0A\00"
// CHECK-NEXT: [[END_STRUCT_U4:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit5.a = private unnamed_addr constant %struct.U5A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U5:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U5A {\0A\00"
// CHECK-NEXT: [[FIELD_U5:@[0-9]+]] = private unnamed_addr constant [10 x i8] c"long a : \00"
// CHECK-NEXT: [[FORMAT_U5:@[0-9]+]] = private unnamed_addr constant [5 x i8] c"%ld\0A\00"
// CHECK-NEXT: [[END_STRUCT_U5:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit6.a = private unnamed_addr constant %struct.U6A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U6:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U6A {\0A\00"
// CHECK-NEXT: [[FIELD_U6:@[0-9]+]] = private unnamed_addr constant [19 x i8] c"unsigned long a : \00"
// CHECK-NEXT: [[FORMAT_U6:@[0-9]+]] = private unnamed_addr constant [5 x i8] c"%lu\0A\00"
// CHECK-NEXT: [[END_STRUCT_U6:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit7.a = private unnamed_addr constant %struct.U7A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U7:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U7A {\0A\00"
// CHECK-NEXT: [[FIELD_U7:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"long long a : \00"
// CHECK-NEXT: [[FORMAT_U7:@[0-9]+]] = private unnamed_addr constant [6 x i8] c"%lld\0A\00"
// CHECK-NEXT: [[END_STRUCT_U7:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit8.a = private unnamed_addr constant %struct.U8A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U8:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U8A {\0A\00"
// CHECK-NEXT: [[FIELD_U8:@[0-9]+]] = private unnamed_addr constant [24 x i8] c"unsigned long long a : \00"
// CHECK-NEXT: [[FORMAT_U8:@[0-9]+]] = private unnamed_addr constant [6 x i8] c"%llu\0A\00"
// CHECK-NEXT: [[END_STRUCT_U8:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit9.a = private unnamed_addr constant %struct.U9A { i8 97 }, align 1
// CHECK-NEXT: [[STRUCT_STR_U9:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U9A {\0A\00"
// CHECK-NEXT: [[FIELD_U9:@[0-9]+]] = private unnamed_addr constant [10 x i8] c"char a : \00"
// CHECK-NEXT: [[FORMAT_U9:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%c\0A\00"
// CHECK-NEXT: [[END_STRUCT_U9:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @.str = private unnamed_addr constant [4 x i8] c"LSE\00", align 1

// CHECK: @__const.unit10.a = private unnamed_addr constant %struct.U10A { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U10:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U10A {\0A\00"
// CHECK-NEXT: [[FIELD_U10:@[0-9]+]] = private unnamed_addr constant [12 x i8] c"char * a : \00"
// CHECK-NEXT: [[FORMAT_U10:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%s\0A\00"
// CHECK-NEXT: [[END_STRUCT_U10:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit11.a = private unnamed_addr constant %struct.U11A { i8* inttoptr (i64 305419896 to i8*) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U11:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U11A {\0A\00"
// CHECK-NEXT: [[FIELD_U11:@[0-9]+]] = private unnamed_addr constant [12 x i8] c"void * a : \00"
// CHECK-NEXT: [[FORMAT_U11:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%p\0A\00"
// CHECK-NEXT: [[END_STRUCT_U11:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit12.a = private unnamed_addr constant %struct.U12A { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U12:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U12A {\0A\00"
// CHECK-NEXT: [[FIELD_U12:@[0-9]+]] = private unnamed_addr constant [18 x i8] c"const char * a : \00"
// CHECK-NEXT: [[FORMAT_U12:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%s\0A\00"
// CHECK-NEXT: [[END_STRUCT_U12:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit13.a = private unnamed_addr constant %struct.U13A { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U13:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U13A {\0A\00"
// CHECK-NEXT: [[FIELD_U13:@[0-9]+]] = private unnamed_addr constant [20 x i8] c"const charstar a : \00"
// CHECK-NEXT: [[FORMAT_U13:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%s\0A\00"
// CHECK-NEXT: [[END_STRUCT_U13:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit14.a = private unnamed_addr constant %struct.U14A { double 0x3FF1F9ACFFA7EB6C }, align 8
// CHECK-NEXT: [[STRUCT_STR_U14:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U14A {\0A\00"
// CHECK-NEXT: [[FIELD_U14:@[0-9]+]] = private unnamed_addr constant [12 x i8] c"double a : \00"
// CHECK-NEXT: [[FORMAT_U14:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%f\0A\00"
// CHECK-NEXT: [[END_STRUCT_U14:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit15.a = private unnamed_addr constant %struct.U15A { [3 x i32] [i32 1, i32 2, i32 3] }, align 4
// CHECK-NEXT: [[STRUCT_STR_U15:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U15A {\0A\00"
// CHECK-NEXT: [[FIELD_U15:@[0-9]+]] = private unnamed_addr constant [12 x i8] c"int[3] a : \00"
// CHECK-NEXT: [[FORMAT_U15:@[0-9]+]] = private unnamed_addr constant [4 x i8] c"%p\0A\00"
// CHECK-NEXT: [[END_STRUCT_U15:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit16.a = private unnamed_addr constant %struct.U16A { i8 12 }, align 1
// CHECK-NEXT: [[STRUCT_STR_U16:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U16A {\0A\00"
// CHECK-NEXT: [[FIELD_U16:@[0-9]+]] = private unnamed_addr constant [13 x i8] c"uint8_t a : \00"
// CHECK-NEXT: [[FORMAT_U16:@[0-9]+]] = private unnamed_addr constant [6 x i8] c"%hhu\0A\00"
// CHECK-NEXT: [[END_STRUCT_U16:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit17.a = private unnamed_addr constant %struct.U17A { i8 12 }, align 1
// CHECK-NEXT: [[STRUCT_STR_U17:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U17A {\0A\00"
// CHECK-NEXT: [[FIELD_U17:@[0-9]+]] = private unnamed_addr constant [12 x i8] c"int8_t a : \00"
// CHECK-NEXT: [[FORMAT_U17:@[0-9]+]] = private unnamed_addr constant [6 x i8] c"%hhd\0A\00"
// CHECK-NEXT: [[END_STRUCT_U17:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

// CHECK: @__const.unit18.a = private unnamed_addr constant %struct.U18A { x86_fp80 0xK3FFF8FCD67FD3F5B6000 }, align 16
// CHECK-NEXT: [[STRUCT_STR_U18:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U18A {\0A\00"
// CHECK-NEXT: [[FIELD_U18:@[0-9]+]] = private unnamed_addr constant [17 x i8] c"long double a : \00"
// CHECK-NEXT: [[FORMAT_U18:@[0-9]+]] = private unnamed_addr constant [5 x i8] c"%Lf\0A\00"
// CHECK-NEXT: [[END_STRUCT_U18:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00"

int printf(const char *fmt, ...) {
    return 0;
}

void unit1() {
  struct U1A {
    short a;
  };

  struct U1A a = {
      .a = 12,
  };
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U1]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U1A, %struct.U1A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* [[FIELD_U1]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i16, i16* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* [[FORMAT_U1]], i32 0, i32 0), i16 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U1]], i32 0, i32 0))
  __builtin_dump_struct(&a, &printf);
}

void unit2() {
  struct U2A {
    unsigned short a;
  };

  struct U2A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U2]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U2A, %struct.U2A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* [[FIELD_U2]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i16, i16* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* [[FORMAT_U2]], i32 0, i32 0), i16 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U2]], i32 0, i32 0))
  __builtin_dump_struct(&a, &printf);
}

void unit3() {
  struct U3A {
    int a;
  };

  struct U3A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U3]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U3A, %struct.U3A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* [[FIELD_U3]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U3]], i32 0, i32 0), i32 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U3]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit4() {
  struct U4A {
    unsigned int a;
  };

  struct U4A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U4]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U4A, %struct.U4A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* [[FIELD_U4]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U4]], i32 0, i32 0), i32 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U4]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit5() {
  struct U5A {
    long a;
  };

  struct U5A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U5]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U5A, %struct.U5A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* [[FIELD_U5]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* [[FORMAT_U5]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U5]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit6() {
  struct U6A {
    unsigned long a;
  };

  struct U6A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U6]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U6A, %struct.U6A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[FIELD_U6]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* [[FORMAT_U6]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U6]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit7() {
  struct U7A {
    long long a;
  };

  struct U7A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U7]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U7A, %struct.U7A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[FIELD_U7]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* [[FORMAT_U7]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U7]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit8() {
  struct U8A {
    unsigned long long a;
  };

  struct U8A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U8]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U8A, %struct.U8A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* [[FIELD_U8]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* [[FORMAT_U8]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U8]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit9() {
  struct U9A {
    char a;
  };

  struct U9A a = {
      .a = 'a',
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U9]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U9A, %struct.U9A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* [[FIELD_U9]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i8, i8* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U9]], i32 0, i32 0), i8 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U9]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit10() {
  struct U10A {
    char *a;
  };

  struct U10A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U10]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U10A, %struct.U10A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* [[FIELD_U10]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U10]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U10]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit11() {
  struct U11A {
    void *a;
  };

  struct U11A a = {
      .a = (void *)0x12345678,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U11]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U11A, %struct.U11A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* [[FIELD_U11]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U11]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U11]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit12() {
  struct U12A {
    const char *a;
  };

  struct U12A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U12]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U12A, %struct.U12A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* [[FIELD_U12]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U12]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U12]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit13() {
  typedef char *charstar;
  struct U13A {
    const charstar a;
  };

  struct U13A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U13]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U13A, %struct.U13A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* [[FIELD_U13]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U13]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U13]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit14() {
  struct U14A {
    double a;
  };

  struct U14A a = {
      .a = 1.123456,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U14]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U14A, %struct.U14A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* [[FIELD_U14]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load double, double* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U14]], i32 0, i32 0), double [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U14]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit15() {
  struct U15A {
    int a[3];
  };

  struct U15A a = {
      .a = {1, 2, 3},
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U15]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U15A, %struct.U15A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* [[FIELD_U15]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load [3 x i32], [3 x i32]* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* [[FORMAT_U15]], i32 0, i32 0), [3 x i32] [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U15]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit16() {
  struct U16A {
    uint8_t a;
  };

  struct U16A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U16]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U16A, %struct.U16A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* [[FIELD_U16]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i8, i8* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* [[FORMAT_U16]], i32 0, i32 0), i8 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U16]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit17() {
  struct U17A {
    int8_t a;
  };

  struct U17A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U17]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U17A, %struct.U17A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* [[FIELD_U17]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load i8, i8* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* [[FORMAT_U17]], i32 0, i32 0), i8 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U17]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit18() {
  struct U18A {
    long double a;
  };

  struct U18A a = {
      .a = 1.123456,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U18]], i32 0, i32 0))
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U18A, %struct.U18A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* [[FIELD_U18]], i32 0, i32 0))
  // CHECK: [[LOAD1:%[0-9]+]] = load x86_fp80, x86_fp80* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* [[FORMAT_U18]], i32 0, i32 0), x86_fp80 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U18]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void test1() {
  struct T1A {
    int a;
    char *b;
  };

  struct T1A a = {
      .a = 12,
      .b = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.T1A, %struct.T1A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: [[RES2:%[0-9]+]] = getelementptr inbounds %struct.T1A, %struct.T1A* %a, i32 0, i32 1
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD2:%[0-9]+]] = load i8*, i8** [[RES2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void test2() {
  struct T2A {
    int a;
  };

  struct T2B {
    int b;
    struct T2A a;
  };

  struct T2B b = {
      .b = 24,
      .a = {
          .a = 12,
      }
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.T2B, %struct.T2B* %b, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: [[NESTED_STRUCT:%[0-9]+]] = getelementptr inbounds %struct.T2B, %struct.T2B* %b, i32 0, i32 1
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES2:%[0-9]+]] = getelementptr inbounds %struct.T2A, %struct.T2A* [[NESTED_STRUCT]], i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD2:%[0-9]+]] = load i32, i32* [[RES2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&b, &printf);
}

void test3() {
  struct T3A {
    union {
      int a;
      char b[4];
    };
  };

  struct T3A a = {
      .a = 42,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.T3A, %struct.T3A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC1:%[0-9]+]] = bitcast %union.anon* [[RES1]] to i32*
  // CHECK: [[LOAD1:%[0-9]+]] = load i32, i32* [[BC1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: [[BC2:%[0-9]+]] = bitcast %union.anon* [[RES1]] to [4 x i8]*
  // CHECK: [[LOAD2:%[0-9]+]] = load [4 x i8], [4 x i8]* [[BC2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, [4 x i8] [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void test4() {
  struct T4A {
    union {
      struct {
        void *a;
      };
      struct {
        unsigned long b;
      };
    };
  };

  struct T4A a = {
      .a = (void *)0x12345678,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.T4A, %struct.T4A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC1:%[0-9]+]] = bitcast %union.anon.0* [[RES1]] to %struct.anon*
  // CHECK: [[RES2:%[0-9]+]] = getelementptr inbounds %struct.anon, %struct.anon* [[BC1]], i32 0, i32 0
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD1]])

  // CHECK: [[BC2:%[0-9]+]] = bitcast %union.anon.0* [[RES1]] to %struct.anon.1*
  // CHECK: [[RES3:%[0-9]+]] = getelementptr inbounds %struct.anon.1, %struct.anon.1* [[BC2]], i32 0, i32 0
  // CHECK: [[LOAD2:%[0-9]+]] = load i64, i64* [[RES3]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i64 [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}
