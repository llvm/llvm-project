// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o %t

void f1() {
  // Scalars in braces.
  int a = { 1 };
}

void f2() {
  int a[2][2] = { { 1, 2 }, { 3, 4 } };
  int b[3][3] = { { 1, 2 }, { 3, 4 } };
  int *c[2] = { &a[1][1], &b[2][2] };
  int *d[2][2] = { {&a[1][1], &b[2][2]}, {&a[0][0], &b[1][1]} };
  int *e[3][3] = { {&a[1][1], &b[2][2]}, {&a[0][0], &b[1][1]} };
  char ext[3][3] = {".Y",".U",".V"};
}

typedef void (* F)(void);
extern void foo(void);
struct S { F f; };
void f3() {
  struct S a[1] = { { foo } };
}

// Constants
// RUN: grep '@g3 = constant i32 10' %t
// RUN: grep '@f4.g4 = internal constant i32 12' %t
const int g3 = 10;
int f4() {
  static const int g4 = 12;
  return g4;
}

// PR6537
typedef union vec3 {
  struct { double x, y, z; };
  double component[3];
} vec3;
vec3 f5(vec3 value) {
  return (vec3) {{
    .x = value.x
  }};
}
