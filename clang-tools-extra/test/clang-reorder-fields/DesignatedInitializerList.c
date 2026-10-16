// RUN: clang-reorder-fields -record-name Foo -fields-order z,w,y,x %s -- | FileCheck %s

struct Foo {
  const int* x; // CHECK:      {{^  double z;}}
  int y;        // CHECK-NEXT: {{^  int w;}}
  double z;     // CHECK-NEXT: {{^  int y;}}
  int w;        // CHECK-NEXT: {{^  const int\* x}}
};

struct Bar {
  char a;
  struct Foo b;
  char c;
};

int main() {
  const int x = 13;
  struct Foo foo1 = { .x=&x, .y=0, .z=1.29, .w=17 }; // CHECK: {{^ struct Foo foo1 = { .z = 1.29, .w = 17, .y = 0, .x = &x };}}
  struct Foo foo2 = { .x=&x, 0, 1.29, 17 };          // CHECK: {{^ struct Foo foo2 = { .z = 1.29, .w = 17, .y = 0, .x = &x };}}
  struct Foo foo3 = { .y=0, .z=1.29, 17, .x=&x };    // CHECK: {{^ struct Foo foo3 = { .z = 1.29, .w = 17, .y = 0, .x = &x };}}
  struct Foo foo4 = { .y=0, .z=1.29, 17 };           // CHECK: {{^ struct Foo foo4 = { .z = 1.29, .w = 17, .y = 0 };}}

  struct Foo foos1[1] = { [0] = {.x=&x, 0, 1.29, 17} };              // CHECK: {{^ struct Foo foos1\[1] = { \[0] = {.z = 1.29, .w = 17, .y = 0, .x = &x} };}}
  struct Foo foos2[1] = { [0].x=&x, [0].y=0, [0].z=1.29, [0].w=17 }; // CHECK: {{^ struct Foo foos2\[1] = { \[0].z = 1.29, \[0].w = 17, \[0].y = 0, \[0].x = &x };}}
  struct Foo foos3[1] = { &x, 0, 1.29, 17 };                         // CHECK: {{^ struct Foo foos3\[1] = { \[0].z = 1.29, \[0].w = 17, \[0].y = 0, \[0].x = &x };}}
  struct Foo foos4[2] = { &x, 0, 1.29, 17, &x, 0, 1.29, 17 };        // CHECK: {{^ struct Foo foos4\[2] = { \[0].z = 1.29, \[0].w = 17, \[0].y = 0, \[0].x = &x, \[1].z = 1.29, \[1].w = 17, \[1].y = 0, \[1].x = &x };}}

  struct Bar bar1 = { .a='a', &x, 0, 1.29, 17, 'c' }; // CHECK: {{^ struct Bar bar1 = { .a = 'a', .b.z = 1.29, .b.w = 17, .b.y = 0, .b.x = &x, .c = 'c' };}}

  return 0;
}
