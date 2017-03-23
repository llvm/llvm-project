// RUN: %clang_cc1 -target-feature +altivec -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// Check initialization

vector int test0 = (vector int)(1);       // CHECK: @test0 = global <4 x i32> <i32 1, i32 1, i32 1, i32 1>
vector float test1 = (vector float)(1.0); // CHECK: @test1 = global <4 x float> <float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}>

// CHECK: @v1 = global <16 x i8> <i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 0, i8 2, i8 0, i8 0, i8 0, i8 3, i8 0, i8 0, i8 0, i8 4>
vector char v1 = (vector char)((vector int)(1, 2, 3, 4));
// CHECK: @v2 = global <16 x i8> <i8 63, i8 -128, i8 0, i8 0, i8 64, i8 0, i8 0, i8 0, i8 64, i8 64, i8 0, i8 0, i8 64, i8 -128, i8 0, i8 0>
vector char v2 = (vector char)((vector float)(1.0f, 2.0f, 3.0f, 4.0f));
// CHECK: @v3 = global <16 x i8> <i8 0, i8 0, i8 0, i8 97, i8 0, i8 0, i8 0, i8 98, i8 0, i8 0, i8 0, i8 99, i8 0, i8 0, i8 0, i8 100>
vector char v3 = (vector char)((vector int)('a', 'b', 'c', 'd'));
// CHECK: @v4 = global <4 x i32> <i32 16909060, i32 0, i32 0, i32 0>
vector int v4 = (vector char){1, 2, 3, 4};

void test2()
{
  vector int vi;
  vector float vf;
  vi = (vector int)(1);             // CHECK: <i32 1, i32 1, i32 1, i32 1>
  vf = (vector float)(1.0);         // CHECK: <float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}>
  vi = (vector int)(1, 2, 3, 4);    // CHECK: <i32 1, i32 2, i32 3, i32 4>
  vi = (vector int)(1, 2, 3, 4, 5); // CHECK: <i32 1, i32 2, i32 3, i32 4>

  vi = (vector int){1};             // CHECK: <i32 1, i32 0, i32 0, i32 0>
  vi = (vector int){1, 2};          // CHECK: <i32 1, i32 2, i32 0, i32 0>
  vi = (vector int){1, 2, 3, 4};    // CHECK: <i32 1, i32 2, i32 3, i32 4>

}

// Check pre/post increment/decrement
void test3() {
  vector int vi;
  vi++;                                    // CHECK: add <4 x i32> {{.*}} <i32 1, i32 1, i32 1, i32 1>
  vector unsigned int vui;
  --vui;                                   // CHECK: add <4 x i32> {{.*}} <i32 -1, i32 -1, i32 -1, i32 -1>
  vector float vf;
  vf++;                                    // CHECK: fadd <4 x float> {{.*}} <float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}, float 1.000000e+{{0+}}>
}
