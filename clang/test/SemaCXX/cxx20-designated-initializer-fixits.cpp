// RUN: %clang_cc1 -std=c++20 %s -Wreorder-init-list -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

// These tests are from clang/test/SemaCXX/cxx2a-initializer-aggregates.cpp
struct C { int :0, x, :0, y, :0; };
C c = {
  .x = 1,
  .x = 1,
  .y = 1,
  .y = 1,
  .x = 1,
  .x = 1,
};
// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:3-[[@LINE-7]]:9}:".x = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:3-[[@LINE-7]]:9}:".x = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:3-[[@LINE-7]]:9}:".x = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:3-[[@LINE-7]]:9}:".x = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:3-[[@LINE-7]]:9}:".y = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:3-[[@LINE-7]]:9}:".y = 1"

namespace GH63605 {
struct A  {
  unsigned x;
  unsigned y;
  unsigned z;
};

struct B {
  unsigned a;
  unsigned b;
};

struct : public A, public B {
  unsigned : 2;
  unsigned a : 6;
  unsigned : 1;
  unsigned b : 6;
  unsigned : 2;
  unsigned c : 6;
  unsigned d : 1;
  unsigned e : 2;
} data = {
  {.z=0,


   .y=1,

   .x=2},
  {.b=3,
   .a=4},
    .e = 1,

    .d = 1,

    .c = 1,
    .b = 1,
    .a = 1,
};
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-17]]:4-[[@LINE-17]]:8}:".x=2"
// CHECK: fix-it:"{{.*}}":{[[@LINE-15]]:4-[[@LINE-15]]:8}:".y=1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-14]]:4-[[@LINE-14]]:8}:".z=0"
// CHECK: fix-it:"{{.*}}":{[[@LINE-14]]:4-[[@LINE-14]]:8}:".a=4"
// CHECK: fix-it:"{{.*}}":{[[@LINE-14]]:4-[[@LINE-14]]:8}:".b=3"
// CHECK: fix-it:"{{.*}}":{[[@LINE-14]]:5-[[@LINE-14]]:11}:".a = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-13]]:5-[[@LINE-13]]:11}:".b = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-12]]:5-[[@LINE-12]]:11}:".c = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-12]]:5-[[@LINE-12]]:11}:".d = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-12]]:5-[[@LINE-12]]:11}:".e = 1"
// END tests from clang/test/SemaCXX/cxx2a-initializer-aggregates.cpp

namespace reorder_derived {
struct col {
  int r;
  int g;
  int b;
};

struct point {
  float x;
  float y;
  float z;
};

struct derived : public col, public point {
  int z2;
  int z1;
};

void test() {
  derived a {
    {.b = 1, .g = 2, .r = 3},
    { .z = 1, .y=2, .x =  3 },
    .z1 = 1,
    .z2 = 2,
  };
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:6-[[@LINE-6]]:12}:".r = 3"
// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:14-[[@LINE-7]]:20}:".g = 2"
// CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:22-[[@LINE-8]]:28}:".b = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:15-[[@LINE-8]]:19}:".y=2"
// CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:21-[[@LINE-9]]:28}:".z = 1"
// CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:7-[[@LINE-10]]:13}:".x =  3"
// CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:5-[[@LINE-10]]:12}:".z2 = 2"
// CHECK: fix-it:"{{.*}}":{[[@LINE-10]]:5-[[@LINE-10]]:12}:".z1 = 1"
} // namespace reorder_derived
