// RUN: %clang_cc1 -emit-llvm -o - -triple i386-linux-gnu %s | FileCheck %s

// This checks that the global won't be marked as common. 
// (It shouldn't because it's being initialized).

int a;
int a = 242;
// CHECK: @a ={{.*}} global i32 242

// This should get normal weak linkage.
int c __attribute__((weak))= 0;
// CHECK: @c = weak{{.*}} global i32 0


// Since this is marked const, it should get weak_odr linkage, since all
// definitions have to be the same.
// CHECK: @d = weak_odr constant i32 0
const int d __attribute__((weak))= 0;

// PR6168 "too many undefs"
struct ManyFields {
  int a;
  int b;
  int c;
  char d;
  int e;
  int f;
};

// CHECK: global %struct.ManyFields { i32 1, i32 2, i32 0, i8 0, i32 0, i32 0 }
struct ManyFields FewInits = {1, 2};


// PR6766
// CHECK: @l ={{.*}} global %struct.K { [6 x i32] [i32 102, i32 111, i32 111, i32 0, i32 0, i32 0], i32 1 }
typedef __WCHAR_TYPE__ wchar_t;
struct K {
  wchar_t L[6];
  int M;
} l =  { { L"foo" }, 1 };


// CHECK: @yuv_types ={{.*}} global [4 x [6 x i8]] {{\[}}[6 x i8] c"4:0:0\00", [6 x i8] c"4:2:0\00", [6 x i8] c"4:2:2\00", [6 x i8] c"4:4:4\00"]
char yuv_types[4][6]= {"4:0:0","4:2:0","4:2:2","4:4:4"};


// NOTE: tentative definitions are processed at the end of the translation unit.

// This shouldn't be emitted as common because it has an explicit section.
// rdar://7119244
// CHECK: @b ={{.*}} global i32 0, section "foo"
int b __attribute__((section("foo")));
