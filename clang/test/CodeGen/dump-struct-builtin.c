// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

#include "Inputs/stdio.h"

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

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U1A, %struct.U1A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i16, i16* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i16 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit2() {
  struct U2A {
    unsigned short a;
  };

  struct U2A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U2A, %struct.U2A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i16, i16* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i16 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit3() {
  struct U3A {
    int a;
  };

  struct U3A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U3A, %struct.U3A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit4() {
  struct U4A {
    unsigned int a;
  };

  struct U4A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U4A, %struct.U4A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i32, i32* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i32 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit5() {
  struct U5A {
    long a;
  };

  struct U5A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U5A, %struct.U5A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit6() {
  struct U6A {
    unsigned long a;
  };

  struct U6A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U6A, %struct.U6A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit7() {
  struct U7A {
    long long a;
  };

  struct U7A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U7A, %struct.U7A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit8() {
  struct U8A {
    long long a;
  };

  struct U8A a = {
      .a = 12,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U8A, %struct.U8A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i64, i64* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i64 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit9() {
  struct U9A {
    char a;
  };

  struct U9A a = {
      .a = 'a',
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U9A, %struct.U9A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i8, i8* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8 [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit10() {
  struct U10A {
    char *a;
  };

  struct U10A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U10A, %struct.U10A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit11() {
  struct U11A {
    void *a;
  };

  struct U11A a = {
      .a = (void *)0x12345678,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U11A, %struct.U11A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit12() {
  struct U12A {
    const char *a;
  };

  struct U12A a = {
      .a = "LSE",
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U12A, %struct.U12A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
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

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U13A, %struct.U13A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load i8*, i8** [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, i8* [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit14() {
  struct U14A {
    double a;
  };

  struct U14A a = {
      .a = 1.123456,
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U14A, %struct.U14A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load double, double* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, double [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
  __builtin_dump_struct(&a, &printf);
}

void unit15() {
  struct U15A {
    int a[3];
  };

  struct U15A a = {
      .a = {1, 2, 3},
  };

  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[RES1:%[0-9]+]] = getelementptr inbounds %struct.U15A, %struct.U15A* %a, i32 0, i32 0
  // CHECK: call i32 (i8*, ...) @printf(
  // CHECK: [[LOAD1:%[0-9]+]] = load [3 x i32], [3 x i32]* [[RES1]],
  // CHECK: call i32 (i8*, ...) @printf({{.*}}, [3 x i32] [[LOAD1]])
  // CHECK: call i32 (i8*, ...) @printf(
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
