// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include "Inputs/stdio.h"
#include <stdint.h>

// CHECK: @__const.unit1.a = private unnamed_addr constant %struct.U1A { i16 12 }, align 2
// CHECK-NEXT: [[STRUCT_STR_U1:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U1A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U1:@[0-9]+]] = private unnamed_addr constant [19 x i8] c"    short a = %hd\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U1:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit2.a = private unnamed_addr constant %struct.U2A { i16 12 }, align 2
// CHECK-NEXT: [[STRUCT_STR_U2:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U2A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U2:@[0-9]+]] = private unnamed_addr constant [28 x i8] c"    unsigned short a = %hu\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U2:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit3.a = private unnamed_addr constant %struct.U3A { i32 12 }, align 4
// CHECK-NEXT: [[STRUCT_STR_U3:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U3A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U3:@[0-9]+]] = private unnamed_addr constant [16 x i8] c"    int a = %d\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U3:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit4.a = private unnamed_addr constant %struct.U4A { i32 12 }, align 4
// CHECK-NEXT: [[STRUCT_STR_U4:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U4A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U4:@[0-9]+]] = private unnamed_addr constant [25 x i8] c"    unsigned int a = %u\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U4:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit5.a = private unnamed_addr constant %struct.U5A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U5:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U5A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U5:@[0-9]+]] = private unnamed_addr constant [18 x i8] c"    long a = %ld\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U5:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit6.a = private unnamed_addr constant %struct.U6A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U6:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U6A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U6:@[0-9]+]] = private unnamed_addr constant [27 x i8] c"    unsigned long a = %lu\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U6:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit7.a = private unnamed_addr constant %struct.U7A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U7:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U7A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U7:@[0-9]+]] = private unnamed_addr constant [24 x i8] c"    long long a = %lld\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U7:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit8.a = private unnamed_addr constant %struct.U8A { i64 12 }, align 8
// CHECK-NEXT: [[STRUCT_STR_U8:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U8A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U8:@[0-9]+]] = private unnamed_addr constant [33 x i8] c"    unsigned long long a = %llu\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U8:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit9.a = private unnamed_addr constant %struct.U9A { i8 97 }, align 1
// CHECK-NEXT: [[STRUCT_STR_U9:@[0-9]+]] = private unnamed_addr constant [14 x i8] c"struct U9A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U9:@[0-9]+]] = private unnamed_addr constant [17 x i8] c"    char a = %c\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U9:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @.str = private unnamed_addr constant [4 x i8] c"LSE\00", align 1
// CHECK: @__const.unit10.a = private unnamed_addr constant %struct.U10A { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U10:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U10A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U10:@[0-9]+]] = private unnamed_addr constant [19 x i8] c"    char * a = %s\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U10:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit11.a = private unnamed_addr constant %struct.U11A { i8* inttoptr (i64 305419896 to i8*) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U11:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U11A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U11:@[0-9]+]] = private unnamed_addr constant [19 x i8] c"    void * a = %p\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U11:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit12.a = private unnamed_addr constant %struct.U12A { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U12:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U12A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U12:@[0-9]+]] = private unnamed_addr constant [25 x i8] c"    const char * a = %s\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U12:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit13.a = private unnamed_addr constant %struct.U13A { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i32 0, i32 0) }, align 8
// CHECK-NEXT: [[STRUCT_STR_U13:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U13A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U13:@[0-9]+]] = private unnamed_addr constant [27 x i8] c"    const charstar a = %s\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U13:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit14.a = private unnamed_addr constant %struct.U14A { double 0x3FF1F9ACFFA7EB6C }, align 8
// CHECK-NEXT: [[STRUCT_STR_U14:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U14A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U14:@[0-9]+]] = private unnamed_addr constant [19 x i8] c"    double a = %f\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U14:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit15.a = private unnamed_addr constant %struct.U15A { [3 x i32] [i32 1, i32 2, i32 3] }, align 4
// CHECK-NEXT: [[STRUCT_STR_U15:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U15A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U15:@[0-9]+]] = private unnamed_addr constant [19 x i8] c"    int[3] a = %p\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U15:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit16.a = private unnamed_addr constant %struct.U16A { i8 12 }, align 1
// CHECK-NEXT: [[STRUCT_STR_U16:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U16A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U16:@[0-9]+]] = private unnamed_addr constant [22 x i8] c"    uint8_t a = %hhu\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U16:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit17.a = private unnamed_addr constant %struct.U17A { i8 12 }, align 1
// CHECK-NEXT: [[STRUCT_STR_U17:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U17A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U17:@[0-9]+]] = private unnamed_addr constant [21 x i8] c"    int8_t a = %hhd\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U17:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

// CHECK: @__const.unit18.a = private unnamed_addr constant %struct.U18A { x86_fp80 0xK3FFF8FCD67FD3F5B6000 }, align 16
// CHECK-NEXT: [[STRUCT_STR_U18:@[0-9]+]] = private unnamed_addr constant [15 x i8] c"struct U18A {\0A\00", align 1
// CHECK-NEXT: [[FIELD_U18:@[0-9]+]] = private unnamed_addr constant [25 x i8] c"    long double a = %Lf\0A\00", align 1
// CHECK-NEXT: [[END_STRUCT_U18:@[0-9]+]] = private unnamed_addr constant [3 x i8] c"}\0A\00", align 1

int printf(const char *fmt, ...) {
    return 0;
}

void unit1(void) {
  struct U1A {
    short a;
  };

  struct U1A a = {
      .a = 12,
  };
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U1]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U1A, %struct.U1A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i16, i16* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[FIELD_U1]], i32 0, i32 0), i16 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U1]], i32 0, i32 0))
  __builtin_dump_struct(&a, &printf);
}

void unit2(void) {
  struct U2A {
    unsigned short a;
  };

  struct U2A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U2]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U2A, %struct.U2A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i16, i16* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* [[FIELD_U2]], i32 0, i32 0), i16 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U2]], i32 0, i32 0))
  __builtin_dump_struct(&a, &printf);
}

void unit3(void) {
  struct U3A {
    int a;
  };

  struct U3A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U3]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U3A, %struct.U3A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* [[FIELD_U3]], i32 0, i32 0), i32 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U3]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit4(void) {
  struct U4A {
    unsigned int a;
  };

  struct U4A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U4]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U4A, %struct.U4A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* [[FIELD_U4]], i32 0, i32 0), i32 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U4]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit5(void) {
  struct U5A {
    long a;
  };

  struct U5A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U5]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U5A, %struct.U5A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* [[FIELD_U5]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U5]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit6(void) {
  struct U6A {
    unsigned long a;
  };

  struct U6A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U6]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U6A, %struct.U6A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* [[FIELD_U6]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U6]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit7(void) {
  struct U7A {
    long long a;
  };

  struct U7A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U7]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U7A, %struct.U7A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* [[FIELD_U7]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U7]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit8(void) {
  struct U8A {
    unsigned long long a;
  };

  struct U8A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U8]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U8A, %struct.U8A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* [[FIELD_U8]], i32 0, i32 0), i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U8]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit9(void) {
  struct U9A {
    char a;
  };

  struct U9A a = {
      .a = 'a',
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* [[STRUCT_STR_U9]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U9A, %struct.U9A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8, i8* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* [[FIELD_U9]], i32 0, i32 0), i8 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U9]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit10(void) {
  struct U10A {
    char *a;
  };

  struct U10A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U10]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U10A, %struct.U10A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[FIELD_U10]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U10]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit11(void) {
  struct U11A {
    void *a;
  };

  struct U11A a = {
      .a = (void *)0x12345678,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U11]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U11A, %struct.U11A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[FIELD_U11]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U11]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit12(void) {
  struct U12A {
    const char *a;
  };

  struct U12A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U12]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U12A, %struct.U12A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* [[FIELD_U12]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U12]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit13(void) {
  typedef char *charstar;
  struct U13A {
    const charstar a;
  };

  struct U13A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U13]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U13A, %struct.U13A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* [[FIELD_U13]], i32 0, i32 0), i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U13]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit14(void) {
  struct U14A {
    double a;
  };

  struct U14A a = {
      .a = 1.123456,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U14]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U14A, %struct.U14A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load double, double* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[FIELD_U14]], i32 0, i32 0), double [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U14]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit15(void) {
  struct U15A {
    int a[3];
  };

  struct U15A a = {
      .a = {1, 2, 3},
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U15]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U15A, %struct.U15A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load [3 x i32], [3 x i32]* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[FIELD_U15]], i32 0, i32 0), [3 x i32] [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U15]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit16(void) {
  struct U16A {
    uint8_t a;
  };

  struct U16A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U16]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U16A, %struct.U16A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8, i8* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* [[FIELD_U16]], i32 0, i32 0), i8 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U16]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit17(void) {
  struct U17A {
    int8_t a;
  };

  struct U17A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U17]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U17A, %struct.U17A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8, i8* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* [[FIELD_U17]], i32 0, i32 0), i8 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U17]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void unit18(void) {
  struct U18A {
    long double a;
  };

  struct U18A a = {
      .a = 1.123456,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* [[STRUCT_STR_U18]], i32 0, i32 0))
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.U18A, %struct.U18A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load x86_fp80, x86_fp80* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* [[FIELD_U18]], i32 0, i32 0), x86_fp80 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* [[END_STRUCT_U18]], i32 0, i32 0)
  __builtin_dump_struct(&a, &printf);
}

void test1(void) {
  struct T1A {
    int a;
    char *b;
  };

  struct T1A a = {
      .a = 12,
      .b = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.T1A, %struct.T1A* %a, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: [[RES2:%.*]] = getelementptr inbounds %struct.T1A, %struct.T1A* %a, i32 0, i32 1
  // CHECK: [[LOAD2:%.*]] = load i8*, i8** [[RES2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void test2(void) {
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
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.T2B, %struct.T2B* %b, i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: [[NESTED_STRUCT:%.*]] = getelementptr inbounds %struct.T2B, %struct.T2B* %b, i32 0, i32 1
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES2:%.*]] = getelementptr inbounds %struct.T2A, %struct.T2A* [[NESTED_STRUCT]], i32 0, i32 0
  // CHECK: [[LOAD2:%.*]] = load i32, i32* [[RES2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&b, &printf);
}

void test3(void) {
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
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.T3A, %struct.T3A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC1:%.*]] = bitcast %union.anon* [[RES1]] to i32*
  // CHECK: [[LOAD1:%.*]] = load i32, i32* [[BC1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: [[BC2:%.*]] = bitcast %union.anon* [[RES1]] to [4 x i8]*
  // CHECK: [[LOAD2:%.*]] = load [4 x i8], [4 x i8]* [[BC2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, [4 x i8] [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void test4(void) {
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
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.T4A, %struct.T4A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC1:%.*]] = bitcast %union.anon.0* [[RES1]] to %struct.anon*
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES2:%.*]] = getelementptr inbounds %struct.anon, %struct.anon* [[BC1]], i32 0, i32 0
  // CHECK: [[LOAD1:%.*]] = load i8*, i8** [[RES2]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC2:%.*]] = bitcast %union.anon.0* [[RES1]] to %struct.anon.1*
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES3:%.*]] = getelementptr inbounds %struct.anon.1, %struct.anon.1* [[BC2]], i32 0, i32 0
  // CHECK: [[LOAD2:%.*]] = load i64, i64* [[RES3]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i64 [[LOAD2]])
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void test5(void) {
  struct T5A {
    unsigned a : 1;
  };

  struct T5A a = {
    .a = 0,
  };

  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* {{.*}}, i32 0, i32 0))
  // CHECK: [[BC1:%.*]] = bitcast %struct.T5A* %a to i8*
  // CHECK: [[LOAD1:%.*]] = load i8, i8* [[BC1]],
  // CHECK: [[CLEAR1:%.*]] = and i8 [[LOAD1]], 1
  // CHECK: [[CAST1:%.*]] = zext i8 [[CLEAR1]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void test6(void) {
  struct T6A {
    unsigned a : 1;
    unsigned b : 1;
    unsigned c : 1;
  };

  struct T6A a = {
    .a = 1,
    .b = 0,
    .c = 1,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC1:%.*]] = bitcast %struct.T6A* %a to i8*
  // CHECK: [[LOAD1:%.*]] = load i8, i8* [[BC1]],
  // CHECK: [[CLEAR1:%.*]] = and i8 [[LOAD1]], 1
  // CHECK: [[CAST1:%.*]] = zext i8 [[CLEAR1]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST1]])
  // CHECK: [[BC2:%.*]] = bitcast %struct.T6A* %a to i8*
  // CHECK: [[LOAD2:%.*]] = load i8, i8* [[BC2]], align 4
  // CHECK: [[LSHR2:%.*]] = lshr i8 [[LOAD2]], 1
  // CHECK: [[CLEAR2:%.*]] = and i8 [[LSHR2]], 1
  // CHECK: [[CAST2:%.*]] = zext i8 [[CLEAR2]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST2]])
  // CHECK: [[BC3:%.*]] = bitcast %struct.T6A* %a to i8*
  // CHECK: [[LOAD3:%.*]] = load i8, i8* [[BC3]], align 4
  // CHECK: [[LSHR3:%.*]] = lshr i8 [[LOAD3]], 2
  // CHECK: [[CLEAR3:%.*]] = and i8 [[LSHR3]], 1
  // CHECK: [[CAST3:%.*]] = zext i8 [[CLEAR3]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST3]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void test7(void) {

  struct T7A {
    unsigned a : 1;
  };

  struct T7B {
    struct T7A a;
    unsigned b : 1;
  };

  struct T7B a = {
    .a = {.a = 0},
    .b = 1,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%.*]] = getelementptr inbounds %struct.T7B, %struct.T7B* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC1:%.*]] = bitcast %struct.T7A* [[RES1]] to i8*
  // CHECK: [[LOAD1:%.*]] = load i8, i8* [[BC1]],
  // CHECK: [[CLEAR1:%.*]] = and i8 [[LOAD1]], 1
  // CHECK: [[CAST1:%.*]] = zext i8 [[CLEAR1]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST1]])
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES2:%.*]] = getelementptr inbounds %struct.T7B, %struct.T7B* %a, i32 0, i32 1
  // CHECK: [[LOAD2:%.*]] = load i8, i8* [[RES2]], align 4
  // CHECK: [[CLEAR2:%.*]] = and i8 [[LOAD2]], 1
  // CHECK: [[CAST2:%.*]] = zext i8 [[CLEAR2]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST2]])
  // CHECK: call i32 (i8*, ...) @printf(
   __builtin_dump_struct(&a, &printf);
}

void test8(void) {
  struct T8A {
    unsigned c : 1;
    unsigned : 3;
    unsigned : 0;
    unsigned b;
  };

  struct T8A a = {
    .b = 2022,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[BC1:%.*]] = bitcast %struct.T8A* %a to i8*
  // CHECK: [[LOAD1:%.*]] = load i8, i8* [[BC1]],
  // CHECK: [[CLEAR1:%.*]] = and i8 [[LOAD1]], 1
  // CHECK: [[CAST1:%.*]] = zext i8 [[CLEAR1]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST1]])
  // CHECK: [[BC2:%.*]] = bitcast %struct.T8A* %a to i8*
  // CHECK: [[LOAD2:%.*]] = load i8, i8* [[BC2]],
  // CHECK: [[LSHR2:%.*]] = lshr i8 [[LOAD2]], 1
  // CHECK: [[CLEAR2:%.*]] = and i8 [[LSHR2]], 7
  // CHECK: [[CAST2:%.*]] = zext i8 [[CLEAR2]] to i32
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* {{.*}}, i32 0, i32 0), i32 [[CAST2]])
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES3:%.*]] = getelementptr inbounds %struct.T8A, %struct.T8A* %a, i32 0, i32 1
  // CHECK: [[LOAD3:%.*]] = load i32, i32* [[RES3]],
  // CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* {{.*}}, i32 0, i32 0), i32 [[LOAD3]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}
