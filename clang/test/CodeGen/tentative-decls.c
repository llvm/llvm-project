// RUN: %clang_cc1 -emit-llvm -w -o - %s | FileCheck %s

// CHECK-DAG: @r = {{(dso_local )?}}global [1 x {{.*}}] zeroinitializer

int r[];
int (*a)[] = &r;

struct s0;
struct s0 x;
// CHECK-DAG: @x = {{(dso_local )?}}global %struct.s0 zeroinitializer

struct s0 y;
// CHECK-DAG: @y = {{(dso_local )?}}global %struct.s0 zeroinitializer
struct s0 *f0() {
  return &y;
}

struct s0 {
  int x;
};

// CHECK-DAG: @b = {{(dso_local )?}}global [1 x {{.*}}] zeroinitializer
int b[];
int *f1() {
  return b;
}

// Check that the most recent tentative definition wins.
// CHECK-DAG: @c = {{(dso_local )?}}global [4 x {{.*}}] zeroinitializer
int c[];
int c[4];

// Check that we emit static tentative definitions
// CHECK-DAG: @c5 = internal global [1 x {{.*}}] zeroinitializer
static int c5[];
static int func() { return c5[0]; }
int callfunc() { return func(); }

