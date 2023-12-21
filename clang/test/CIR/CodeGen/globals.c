// There seems to be some differences in how constant expressions are evaluated
// in C vs C++. This causees the code gen for C initialized globals to be a
// bit different from the C++ version. This test ensures that these differences
// are accounted for.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

char string[] = "whatnow";
// CHECK: cir.global external @string = #cir.const_array<"whatnow\00" : !cir.array<!s8i x 8>> : !cir.array<!s8i x 8>
int sint[] = {123, 456, 789};
// CHECK: cir.global external @sint = #cir.const_array<[#cir.int<123> : !s32i, #cir.int<456> : !s32i, #cir.int<789> : !s32i]> : !cir.array<!s32i x 3>
int filler_sint[4] = {1, 2}; // Ensure missing elements are zero-initialized.
// CHECK: cir.global external @filler_sint = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 4>
int excess_sint[2] = {1, 2, 3, 4}; // Ensure excess elements are ignored.
// CHECK: cir.global external @excess_sint = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>
float flt[] = {1.0, 2.0};
// CHECK: cir.global external @flt = #cir.const_array<[1.000000e+00 : f32, 2.000000e+00 : f32]> : !cir.array<f32 x 2>

// Tentative definition is just a declaration.
int tentativeB;
int tentativeB = 1;
// CHECK: cir.global external @tentativeB = #cir.int<1> : !s32i

// Tentative incomplete definition is just a declaration.
int tentativeE[];
int tentativeE[2] = {1, 2};
// CHECK: cir.global external @tentativeE = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>

int twoDim[2][2] = {{1, 2}, {3, 4}};
// CHECK: cir.global external @twoDim = #cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 2>]> : !cir.array<!cir.array<!s32i x 2> x 2>

struct {
  int x;
  int y[2][2];
} nestedTwoDim = {1, {{2, 3}, {4, 5}}};
// CHECK: cir.global external @nestedTwoDim = #cir.const_struct<{#cir.int<1> : !s32i, #cir.const_array<[#cir.const_array<[#cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<4> : !s32i, #cir.int<5> : !s32i]> : !cir.array<!s32i x 2>]> : !cir.array<!cir.array<!s32i x 2> x 2>}>

struct {
  char x[3];
  char y[3];
  char z[3];
} nestedString = {"1", "", "\0"};
// CHECK: cir.global external @nestedString = #cir.const_struct<{#cir.const_array<"1\00\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>, #cir.const_array<"\00\00\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>, #cir.const_array<"\00\00\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>}>

struct {
  char *name;
} nestedStringPtr = {"1"};
// CHECK: cir.global external @nestedStringPtr = #cir.const_struct<{#cir.global_view<@".str"> : !cir.ptr<!s8i>}>

int *globalPtr = &nestedString.y[1];
// CHECK: cir.global external @globalPtr = #cir.global_view<@nestedString, [1 : i32, 1 : i32]> : !cir.ptr<!s32i>

const int i = 12;
int i2 = i;
struct { int i; } i3 = {i};
// CHECK: cir.global external @i2 = #cir.int<12> : !s32i
// CHECK: cir.global external @i3 = #cir.const_struct<{#cir.int<12> : !s32i}> : !ty_22anon2E722

int a[10][10][10];
int *a2 = &a[3][0][8];
struct { int *p; } a3 = {&a[3][0][8]};
// CHECK: cir.global external @a2 = #cir.global_view<@a, [3 : i32, 0 : i32, 8 : i32]> : !cir.ptr<!s32i>
// CHECK: cir.global external @a3 = #cir.const_struct<{#cir.global_view<@a, [3 : i32, 0 : i32, 8 : i32]> : !cir.ptr<!s32i>}> : !ty_22anon2E922

int p[10];
int *p1 = &p[0];
struct { int *x; } p2 = {&p[0]};
// CHECK: cir.global external @p1 = #cir.global_view<@p> : !cir.ptr<!s32i>
// CHECK: cir.global external @p2 = #cir.const_struct<{#cir.global_view<@p> : !cir.ptr<!s32i>}> : !ty_22anon2E1122

int q[10];
int *q1 = q;
struct { int *x; } q2 = {q};
// CHECK: cir.global external @q1 = #cir.global_view<@q> : !cir.ptr<!s32i>
// CHECK: cir.global external @q2 = #cir.const_struct<{#cir.global_view<@q> : !cir.ptr<!s32i>}> : !ty_22anon2E1322

int foo() {
    extern int optind;
    return optind;
}
// CHECK: cir.global "private" external @optind : !s32i
// CHECK: cir.func {{.*@foo}}
// CHECK:   {{.*}} = cir.get_global @optind : cir.ptr <!s32i>

// TODO: test tentatives with internal linkage.

// Tentative definition is THE definition. Should be zero-initialized.
int tentativeA;
float tentativeC;
int tentativeD[];
float zeroInitFlt[2];
// CHECK: cir.global external @tentativeA = #cir.int<0> : !s32i
// CHECK: cir.global external @tentativeC = 0.000000e+00 : f32
// CHECK: cir.global external @tentativeD = #cir.zero : !cir.array<!s32i x 1>
// CHECK: cir.global external @zeroInitFlt = #cir.zero : !cir.array<f32 x 2>
