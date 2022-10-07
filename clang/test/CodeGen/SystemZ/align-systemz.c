// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm %s -o - | FileCheck %s

// SystemZ prefers to align all global variables to two bytes.

struct test {
   signed char a;
};

char c;
// CHECK-DAG: @c ={{.*}} global i8 0, align 2

struct test s;
// CHECK-DAG: @s ={{.*}} global %struct.test zeroinitializer, align 2

extern char ec;
// CHECK-DAG: @ec = external global i8, align 2

extern struct test es;
// CHECK-DAG: @es = external global %struct.test, align 2

// Dummy function to make sure external symbols are used.
void func (void)
{
  c = ec;
  s = es;
}


// The SystemZ ABI aligns __int128_t to only eight bytes.

struct S_int128 {  __int128_t B; } Obj_I128;
__int128_t GlobI128;
// CHECK: @Obj_I128 = global %struct.S_int128 zeroinitializer, align 8
// CHECK: @GlobI128 = global i128 0, align 8


// Alignment should be respected for coerced argument loads

struct arg { long y __attribute__((packed, aligned(4))); };

extern struct arg x;
void f(struct arg);

void test (void)
{
  f(x);
}

// CHECK-LABEL: @test
// CHECK: load i64, ptr @x, align 4
