// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes \
// RUN:   -emit-llvm -finclude-default-header -o - %s | FileCheck %s --check-prefix=CHECK,COL-CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes \
// RUN:   -emit-llvm -finclude-default-header -fmatrix-memory-layout=row-major -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK,ROW-CHECK

// Verify that matrix initializer lists store elements in the correct memory
// layout. The initializer list {1,2,3,4,5,6} for a float2x3 (2 rows, 3 cols)
// is in row-major order: row0=[1,2,3], row1=[4,5,6].
//
// With column-major (default) memory layout, the stored vector should be
// reordered to: col0=[1,4], col1=[2,5], col2=[3,6] = <1,4,2,5,3,6>.
//
// With row-major memory layout, the stored vector stays as-is: <1,2,3,4,5,6>.

export float test_row0_col2() {
// CHECK-LABEL: define {{.*}} float @_Z14test_row0_col2v
// COL-CHECK: store <6 x float> <float 1.000000e+00, float 4.000000e+00, float 2.000000e+00, float 5.000000e+00, float 3.000000e+00, float 6.000000e+00>
// COL-CHECK: extractelement <6 x float> %{{.*}}, i32 4
// ROW-CHECK: store <6 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00>
// ROW-CHECK: extractelement <6 x float> %{{.*}}, i32 2
  float2x3 M = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  // Row 0, Col 2 in row-major is the 3rd element = 3.0
  return M[0][2];
}

export float test_row1_col0() {
// CHECK-LABEL: define {{.*}} float @_Z14test_row1_col0v
// COL-CHECK: store <6 x float> <float 1.000000e+00, float 4.000000e+00, float 2.000000e+00, float 5.000000e+00, float 3.000000e+00, float 6.000000e+00>
// COL-CHECK: extractelement <6 x float> %{{.*}}, i32 1
// ROW-CHECK: store <6 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 6.000000e+00>
// ROW-CHECK: extractelement <6 x float> %{{.*}}, i32 3
  float2x3 M = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  // Row 1, Col 0 in row-major is the 4th element = 4.0
  return M[1][0];
}

// Verify the shuffle is emitted for non-constant init lists when the memory
// layout is column-major, and not emitted when it is row-major.

export float2x3 test_dynamic(float a, float b, float c,
                             float d, float e, float f) {
// CHECK-LABEL: define {{.*}} <6 x float> @_Z12test_dynamicffffff
// CHECK: [[A:%.*]] = load float, ptr %a.addr
// CHECK: [[VECINIT0:%.*]] = insertelement <6 x float> poison, float [[A]], i32 0
// CHECK: [[B:%.*]] = load float, ptr %b.addr
// CHECK: [[VECINIT1:%.*]] = insertelement <6 x float> [[VECINIT0]], float [[B]], i32 1
// CHECK: [[C:%.*]] = load float, ptr %c.addr
// CHECK: [[VECINIT2:%.*]] = insertelement <6 x float> [[VECINIT1]], float [[C]], i32 2
// CHECK: [[D:%.*]] = load float, ptr %d.addr
// CHECK: [[VECINIT3:%.*]] = insertelement <6 x float> [[VECINIT2]], float [[D]], i32 3
// CHECK: [[E:%.*]] = load float, ptr %e.addr
// CHECK: [[VECINIT4:%.*]] = insertelement <6 x float> [[VECINIT3]], float [[E]], i32 4
// CHECK: [[F:%.*]] = load float, ptr %f.addr
// CHECK: [[VECINIT5:%.*]] = insertelement <6 x float> [[VECINIT4]], float [[F]], i32 5
// COL-CHECK: shufflevector <6 x float> [[VECINIT5]], <6 x float> poison, <6 x i32> <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
// ROW-CHECK-NOT: shufflevector
// ROW-CHECK: store <6 x float> [[VECINIT5]], ptr
  return (float2x3){a, b, c, d, e, f};
}
