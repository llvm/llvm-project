// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgpu-amd-amdhsa -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgpu-amd-amdhsa -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY %s
// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Array shaping expressions used as the base of array sections in the motion
// clauses of 'target update' (OpenMP 5.0 [2.1.4, Array Shaping]). A column of
// the shaped two-dimensional array is a non-contiguous section; it has to be
// transferred through a descriptor with one entry per dimension, exactly like
// the same section of a declared array.

// FROM | NON_CONTIG, one entry per list item; the size slot holds the number
// of dimensions of the descriptor (two array dimensions plus the element).
// CHECK-DAG: [[FROM2_SIZES:@.+]] = private unnamed_addr constant [2 x i64] [i64 3, i64 3]
// CHECK-DAG: [[FROM2_TYPES:@.+]] = private unnamed_addr constant [2 x i64] [i64 17592186044418, i64 17592186044418]
// TO | NON_CONTIG
// CHECK-DAG: [[TO2_SIZES:@.+]] = private unnamed_addr constant [2 x i64] [i64 3, i64 3]
// CHECK-DAG: [[TO2_TYPES:@.+]] = private unnamed_addr constant [2 x i64] [i64 17592186044417, i64 17592186044417]
// The declared variable length array control.
// CHECK-DAG: [[VLA_SIZES:@.+]] = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-DAG: [[VLA_TYPES:@.+]] = private unnamed_addr constant [1 x i64] [i64 17592186044418]
// Constant dimensions.
// CHECK-DAG: [[CONST_SIZES:@.+]] = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-DAG: [[CONST_TYPES:@.+]] = private unnamed_addr constant [1 x i64] [i64 17592186044418]
// A row of the shaped array is contiguous: 4 doubles, TO.
// CHECK-DAG: [[ROW_SIZES:@.+]] = private unnamed_addr constant [1 x i64] [i64 32]
// CHECK-DAG: [[ROW_TYPES:@.+]] = private unnamed_addr constant [1 x i64] [i64 1]
// Shaped pointers that are a member, a global and an array element: one entry
// with the three descriptor dimensions each.
// CHECK-DAG: [[MEMBER_SIZES:@.+]] = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-DAG: [[MEMBER_TYPES:@.+]] = private unnamed_addr constant [1 x i64] [i64 17592186044418]
// CHECK-DAG: [[GLOBAL_SIZES:@.+]] = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-DAG: [[GLOBAL_TYPES:@.+]] = private unnamed_addr constant [1 x i64] [i64 17592186044418]
// CHECK-DAG: [[ELEMENT_SIZES:@.+]] = private unnamed_addr constant [1 x i64] [i64 3]
// CHECK-DAG: [[ELEMENT_TYPES:@.+]] = private unnamed_addr constant [1 x i64] [i64 17592186044417]
// Combined struct entry, two members MEMBER_OF(1) | TO, section TO | NON_CONTIG.
// CHECK-DAG: [[MIXED_TYPES:@.+]] = private unnamed_addr constant [4 x i64] [i64 0, i64 281474976710657, i64 281474976710657, i64 17592186044417]

void exch_data(double *a, int nx, int ny);

// The example of the OpenMP Examples document: the boundary columns of a
// pointer shaped as a (nx) x (ny+2) array.
// CHECK-LABEL: define {{.*}}void @array_ing(
// CHECK:       store ptr %a, ptr [[A_ADDR:%.+]],
// CHECK-NEXT:  store i32 %nx, ptr [[NX_ADDR:%.+]],
// CHECK-NEXT:  store i32 %ny, ptr [[NY_ADDR:%.+]],
void array_ing(double *a, int nx, int ny) {
#pragma omp target data map(a[0 : nx * (ny + 2)])
  {
    // CHECK:      call void @__tgt_target_data_begin_mapper(

    // The base pointer is the value of 'a'; the dimensions of the shape are
    // evaluated like the sizes of a variable length array.
    // CHECK:      [[BP:%.+]] = load ptr, ptr [[A_ADDR]],
    // CHECK-NEXT: [[NX:%.+]] = load i32, ptr [[NX_ADDR]],
    // CHECK-NEXT: [[NX_64:%.+]] = zext i32 [[NX]] to i64
    // CHECK-NEXT: [[NY:%.+]] = load i32, ptr [[NY_ADDR]],
    // CHECK-NEXT: [[NY2:%.+]] = add nsw i32 [[NY]], 2
    // CHECK-NEXT: [[NY2_64:%.+]] = zext i32 [[NY2]] to i64
    // The section pointer is &a[0][1] of the shaped array.
    // CHECK-NEXT: [[A:%.+]] = load ptr, ptr [[A_ADDR]],
    // CHECK-NEXT: [[ROW0:%.+]] = mul nsw i64 0, [[NY2_64]]
    // CHECK-NEXT: [[ROW0_PTR:%.+]] = getelementptr inbounds nuw double, ptr [[A]], i64 [[ROW0]]
    // CHECK-NEXT: [[COL1_PTR:%.+]] = getelementptr inbounds nuw double, ptr [[ROW0_PTR]], i64 1
    // Descriptor of the outer dimension: offset 0, count nx, stride (ny+2)*8.
    // CHECK-NEXT: [[NY_B:%.+]] = load i32, ptr [[NY_ADDR]],
    // CHECK-NEXT: [[NY2_B:%.+]] = add nsw i32 [[NY_B]], 2
    // CHECK-NEXT: [[NY2_B64:%.+]] = zext i32 [[NY2_B]] to i64
    // CHECK-NEXT: [[NX_B:%.+]] = load i32, ptr [[NX_ADDR]],
    // CHECK-NEXT: [[COUNT0:%.+]] = zext i32 [[NX_B]] to i64
    // CHECK-NEXT: [[STRIDE0:%.+]] = mul nuw i64 8, [[NY2_B64]]
    // CHECK-NEXT: [[OFFSET0:%.+]] = mul nuw i64 [[STRIDE0]], 0

    // The second list item, &a[0][ny] of the shaped array; its own shaping
    // expression evaluates the dimensions again.
    // CHECK-NEXT: [[BP2:%.+]] = load ptr, ptr [[A_ADDR]],
    // CHECK-NEXT: [[NY_C:%.+]] = load i32, ptr [[NY_ADDR]],
    // CHECK-NEXT: [[NY_IDX:%.+]] = sext i32 [[NY_C]] to i64
    // CHECK-NEXT: [[NX_C:%.+]] = load i32, ptr [[NX_ADDR]],
    // CHECK-NEXT: [[NX_C64:%.+]] = zext i32 [[NX_C]] to i64
    // CHECK-NEXT: [[NY_D:%.+]] = load i32, ptr [[NY_ADDR]],
    // CHECK-NEXT: [[NY2_D:%.+]] = add nsw i32 [[NY_D]], 2
    // CHECK-NEXT: [[NY2_D64:%.+]] = zext i32 [[NY2_D]] to i64
    // CHECK-NEXT: [[A2:%.+]] = load ptr, ptr [[A_ADDR]],
    // CHECK-NEXT: [[ROW0_2:%.+]] = mul nsw i64 0, [[NY2_D64]]
    // CHECK-NEXT: [[ROW0_PTR2:%.+]] = getelementptr inbounds nuw double, ptr [[A2]], i64 [[ROW0_2]]
    // CHECK-NEXT: [[COLNY_PTR:%.+]] = getelementptr inbounds nuw double, ptr [[ROW0_PTR2]], i64 [[NY_IDX]]
    // Its inner dimension starts at offset ny*8.
    // CHECK-NEXT: [[NY_E:%.+]] = load i32, ptr [[NY_ADDR]],
    // CHECK-NEXT: [[NY2_E:%.+]] = add nsw i32 [[NY_E]], 2
    // CHECK-NEXT: [[NY2_E64:%.+]] = zext i32 [[NY2_E]] to i64
    // CHECK-NEXT: [[NY_F:%.+]] = load i32, ptr [[NY_ADDR]],
    // CHECK-NEXT: [[NY_F64:%.+]] = zext i32 [[NY_F]] to i64
    // CHECK-NEXT: [[OFFSET1_2:%.+]] = mul nuw i64 8, [[NY_F64]]
    // CHECK-NEXT: [[NX_G:%.+]] = load i32, ptr [[NX_ADDR]],
    // CHECK-NEXT: [[COUNT0_2:%.+]] = zext i32 [[NX_G]] to i64
    // CHECK-NEXT: [[STRIDE0_2:%.+]] = mul nuw i64 8, [[NY2_E64]]
    // CHECK-NEXT: [[OFFSET0_2:%.+]] = mul nuw i64 [[STRIDE0_2]], 0

    // CHECK:      store ptr [[BP]], ptr
    // CHECK:      store ptr [[COL1_PTR]], ptr
    // CHECK:      store ptr [[BP2]], ptr
    // CHECK:      store ptr [[COLNY_PTR]], ptr

    // Descriptor of the first list item: [0:nx] x [1] x element.
    // CHECK:      [[DIM0:%.+]] = getelementptr inbounds [3 x [[DESC:%struct.descriptor_dim[.0-9]*]]], ptr [[DIMS:%.+]], i64 0, i64 0
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM0]], i32 0, i32 0
    // CHECK-NEXT: store i64 [[OFFSET0]], ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM0]], i32 0, i32 1
    // CHECK-NEXT: store i64 [[COUNT0]], ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM0]], i32 0, i32 2
    // CHECK-NEXT: store i64 [[STRIDE0]], ptr [[F]],
    // CHECK-NEXT: [[DIM1:%.+]] = getelementptr inbounds [3 x [[DESC]]], ptr [[DIMS]], i64 0, i64 1
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM1]], i32 0, i32 0
    // CHECK-NEXT: store i64 8, ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM1]], i32 0, i32 1
    // CHECK-NEXT: store i64 1, ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM1]], i32 0, i32 2
    // CHECK-NEXT: store i64 8, ptr [[F]],
    // CHECK-NEXT: [[DIM2:%.+]] = getelementptr inbounds [3 x [[DESC]]], ptr [[DIMS]], i64 0, i64 2
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM2]], i32 0, i32 0
    // CHECK-NEXT: store i64 0, ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM2]], i32 0, i32 1
    // CHECK-NEXT: store i64 8, ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM2]], i32 0, i32 2
    // CHECK-NEXT: store i64 1, ptr [[F]],
    // CHECK-NEXT: [[PTR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS:%.+]], i32 0, i32 0
    // CHECK-NEXT: store ptr [[DIMS]], ptr [[PTR0]],

    // Descriptor of the second list item: [0:nx] x [ny] x element.
    // CHECK-NEXT: [[DIM0_2:%.+]] = getelementptr inbounds [3 x [[DESC]]], ptr [[DIMS2:%.+]], i64 0, i64 0
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM0_2]], i32 0, i32 0
    // CHECK-NEXT: store i64 [[OFFSET0_2]], ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM0_2]], i32 0, i32 1
    // CHECK-NEXT: store i64 [[COUNT0_2]], ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM0_2]], i32 0, i32 2
    // CHECK-NEXT: store i64 [[STRIDE0_2]], ptr [[F]],
    // CHECK-NEXT: [[DIM1_2:%.+]] = getelementptr inbounds [3 x [[DESC]]], ptr [[DIMS2]], i64 0, i64 1
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM1_2]], i32 0, i32 0
    // CHECK-NEXT: store i64 [[OFFSET1_2]], ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM1_2]], i32 0, i32 1
    // CHECK-NEXT: store i64 1, ptr [[F]],
    // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[DESC]], ptr [[DIM1_2]], i32 0, i32 2
    // CHECK-NEXT: store i64 8, ptr [[F]],
    // CHECK:      [[PTR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[PTRS]], i32 0, i32 1
    // CHECK-NEXT: store ptr [[DIMS2]], ptr [[PTR1]],
    // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr %{{.+}}, ptr %{{.+}}, ptr [[FROM2_SIZES]], ptr [[FROM2_TYPES]], ptr null, ptr null)
#pragma omp target update from((([nx][ny + 2])a)[0:nx][1], (([nx][ny + 2])a)[0:nx][ny])

    // CHECK:      call void @exch_data(
    exch_data(a, nx, ny);

    // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr %{{.+}}, ptr %{{.+}}, ptr [[TO2_SIZES]], ptr [[TO2_TYPES]], ptr null, ptr null)
#pragma omp target update to((([nx][ny + 2])a)[0:nx][0], (([nx][ny + 2])a)[0:nx][ny + 1])
    // CHECK:      call void @__tgt_target_data_end_mapper(
  }
}

// The same section of a declared variable length array: the dimension size is
// not a constant, so contiguity cannot be decided at compile time and the
// section is transferred through a descriptor as well (one entry, not whole
// rows).
// CHECK-LABEL: define {{.*}}void @vla_control(
// CHECK:       store i32 %nx, ptr [[VNX_ADDR:%.+]],
// CHECK-NEXT:  store i32 %ny, ptr [[VNY_ADDR:%.+]],
void vla_control(int nx, int ny) {
  double c[nx][ny + 2];
  // CHECK:      [[VLA:%.+]] = alloca double, i64
  // CHECK:      [[VROW0:%.+]] = mul nsw i64 0, [[VNY2_64:%.+]]
  // CHECK-NEXT: [[VROW0_PTR:%.+]] = getelementptr inbounds nuw double, ptr [[VLA]], i64 [[VROW0]]
  // CHECK-NEXT: [[VCOL1_PTR:%.+]] = getelementptr inbounds nuw double, ptr [[VROW0_PTR]], i64 1
  // CHECK-NEXT: [[VNY_B:%.+]] = load i32, ptr [[VNY_ADDR]],
  // CHECK-NEXT: [[VNY2_B:%.+]] = add nsw i32 [[VNY_B]], 2
  // CHECK-NEXT: [[VNY2_B64:%.+]] = zext i32 [[VNY2_B]] to i64
  // CHECK-NEXT: [[VNX_B:%.+]] = load i32, ptr [[VNX_ADDR]],
  // CHECK-NEXT: [[VCOUNT0:%.+]] = zext i32 [[VNX_B]] to i64
  // CHECK-NEXT: [[VSTRIDE0:%.+]] = mul nuw i64 8, [[VNY2_B64]]
  // CHECK-NEXT: [[VOFFSET0:%.+]] = mul nuw i64 [[VSTRIDE0]], 0
  // CHECK:      store ptr [[VLA]], ptr
  // CHECK:      store ptr [[VCOL1_PTR]], ptr
  // CHECK:      [[VDIM0:%.+]] = getelementptr inbounds [3 x [[VDESC:%struct.descriptor_dim[.0-9]*]]], ptr [[VDIMS:%.+]], i64 0, i64 0
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[VDESC]], ptr [[VDIM0]], i32 0, i32 0
  // CHECK-NEXT: store i64 [[VOFFSET0]], ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[VDESC]], ptr [[VDIM0]], i32 0, i32 1
  // CHECK-NEXT: store i64 [[VCOUNT0]], ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[VDESC]], ptr [[VDIM0]], i32 0, i32 2
  // CHECK-NEXT: store i64 [[VSTRIDE0]], ptr [[F]],
  // CHECK-NEXT: [[VDIM1:%.+]] = getelementptr inbounds [3 x [[VDESC]]], ptr [[VDIMS]], i64 0, i64 1
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[VDESC]], ptr [[VDIM1]], i32 0, i32 0
  // CHECK-NEXT: store i64 8, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[VDESC]], ptr [[VDIM1]], i32 0, i32 1
  // CHECK-NEXT: store i64 1, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[VDESC]], ptr [[VDIM1]], i32 0, i32 2
  // CHECK-NEXT: store i64 8, ptr [[F]],
  // CHECK:      store ptr [[VDIMS]], ptr
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr [[VLA_SIZES]], ptr [[VLA_TYPES]], ptr null, ptr null)
#pragma omp target update from(c[0:nx][1])
}

// Constant dimensions: the descriptors are compile-time constants.
// CHECK-LABEL: define {{.*}}void @const_dims(
// CHECK:       store ptr %p, ptr [[P_ADDR:%.+]],
void const_dims(double *p) {
  // &p[0][1] of the shaped double[3][4].
  // CHECK:      [[CBP:%.+]] = load ptr, ptr [[P_ADDR]],
  // CHECK-NEXT: [[CP:%.+]] = load ptr, ptr [[P_ADDR]],
  // CHECK-NEXT: [[CROW0:%.+]] = getelementptr inbounds nuw [4 x double], ptr [[CP]], i64 0
  // CHECK-NEXT: [[CDECAY:%.+]] = getelementptr inbounds [4 x double], ptr [[CROW0]], i64 0, i64 0
  // CHECK-NEXT: [[CCOL1:%.+]] = getelementptr inbounds nuw double, ptr [[CDECAY]], i64 1
  // CHECK:      store ptr [[CBP]], ptr
  // CHECK:      store ptr [[CCOL1]], ptr
  // [0:3]: offset 0, count 3, stride 4*8; [1]: offset 8, count 1, stride 8;
  // element: offset 0, count 8 bytes, stride 1.
  // CHECK:      [[CDIM0:%.+]] = getelementptr inbounds [3 x [[CDESC:%struct.descriptor_dim[.0-9]*]]], ptr [[CDIMS:%.+]], i64 0, i64 0
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM0]], i32 0, i32 0
  // CHECK-NEXT: store i64 0, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM0]], i32 0, i32 1
  // CHECK-NEXT: store i64 3, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM0]], i32 0, i32 2
  // CHECK-NEXT: store i64 32, ptr [[F]],
  // CHECK-NEXT: [[CDIM1:%.+]] = getelementptr inbounds [3 x [[CDESC]]], ptr [[CDIMS]], i64 0, i64 1
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM1]], i32 0, i32 0
  // CHECK-NEXT: store i64 8, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM1]], i32 0, i32 1
  // CHECK-NEXT: store i64 1, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM1]], i32 0, i32 2
  // CHECK-NEXT: store i64 8, ptr [[F]],
  // CHECK-NEXT: [[CDIM2:%.+]] = getelementptr inbounds [3 x [[CDESC]]], ptr [[CDIMS]], i64 0, i64 2
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM2]], i32 0, i32 0
  // CHECK-NEXT: store i64 0, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM2]], i32 0, i32 1
  // CHECK-NEXT: store i64 8, ptr [[F]],
  // CHECK-NEXT: [[F:%.+]] = getelementptr inbounds nuw [[CDESC]], ptr [[CDIM2]], i32 0, i32 2
  // CHECK-NEXT: store i64 1, ptr [[F]],
  // CHECK:      store ptr [[CDIMS]], ptr
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr [[CONST_SIZES]], ptr [[CONST_TYPES]], ptr null, ptr null)
#pragma omp target update from((([3][4])p)[0:3][1])

  // A subscript of the shaped array selects one row, which is contiguous:
  // &p[1][0], 4 doubles.
  // CHECK:      [[RBP:%.+]] = load ptr, ptr [[P_ADDR]],
  // CHECK-NEXT: [[RP:%.+]] = load ptr, ptr [[P_ADDR]],
  // CHECK-NEXT: [[RROW1:%.+]] = getelementptr inbounds nuw [4 x double], ptr [[RP]], i64 1
  // CHECK-NEXT: [[RDECAY:%.+]] = getelementptr inbounds [4 x double], ptr [[RROW1]], i64 0, i64 0
  // CHECK-NEXT: [[RCOL0:%.+]] = getelementptr inbounds nuw double, ptr [[RDECAY]], i64 0
  // CHECK:      store ptr [[RBP]], ptr
  // CHECK:      store ptr [[RCOL0]], ptr
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr [[ROW_SIZES]], ptr [[ROW_TYPES]], ptr null, ptr null)
#pragma omp target update to((([3][4])p)[1][0:4])
}

struct S {
  double *p;
};
double *g;

// Other bases of the shaped pointer: a data member, a global variable and an
// element of an array of pointers. The shaped pointer value is the base
// pointer of the entry, and only the sections of the shaped array are
// dimensions of the descriptor.
// CHECK-LABEL: define {{.*}}void @other_bases(
void other_bases(struct S s, double **pp, int n, int m) {
  // CHECK:      [[SP:%.+]] = load ptr, ptr [[SP_ADDR:%.+]],
  // CHECK:      store ptr [[SP]], ptr
  // CHECK:      getelementptr inbounds [3 x %struct.descriptor_dim{{.*}}], ptr %{{.+}}, i64 0, i64 0
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr [[MEMBER_SIZES]], ptr [[MEMBER_TYPES]], ptr null, ptr null)
#pragma omp target update from((([n][m])s.p)[0:n][1])
  // CHECK:      [[G:%.+]] = load ptr, ptr @g,
  // CHECK:      store ptr [[G]], ptr
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr [[GLOBAL_SIZES]], ptr [[GLOBAL_TYPES]], ptr null, ptr null)
#pragma omp target update from((([n][m])g)[0:n][1])
  // CHECK:      [[PP3_ADDR:%.+]] = getelementptr inbounds ptr, ptr %{{.+}}, i64 3
  // CHECK-NEXT: [[PP3:%.+]] = load ptr, ptr [[PP3_ADDR]],
  // CHECK:      store ptr [[PP3]], ptr
  // CHECK:      getelementptr inbounds [3 x %struct.descriptor_dim{{.*}}], ptr %{{.+}}, i64 0, i64 0
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr [[ELEMENT_SIZES]], ptr [[ELEMENT_TYPES]], ptr null, ptr null)
#pragma omp target update to((([n][m])pp[3])[0:n][1])
  // A contiguous section keeps the pointer-attachment path: the member is the
  // base pointer there as well.
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr @{{.+}}, ptr null, ptr null)
#pragma omp target update to((([n])s.p)[1:n - 2])
}

// Other clauses of the directive, and the other clauses that allow array
// shaping (depend, affinity) with a shaped base. With 'nowait' the update is
// issued from an outlined target task.
// CHECK-LABEL: define {{.*}}void @clauses(
void clauses(double *p, int n, int m, int dev) {
  // CHECK-DAG:  call ptr @__kmpc_omp_target_task_alloc(
  // CHECK-DAG:  call i32 @__kmpc_omp_reg_task_with_affinity(
  // CHECK-DAG:  call i32 @__kmpc_omp_task_with_deps(
  // CHECK-DAG:  call void @__tgt_target_data_update_nowait_mapper(ptr @{{.+}}, i64 %{{.+}}, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr @{{.+}}, ptr null, ptr null, i32 0, ptr null, i32 0, ptr null)
#pragma omp target update to((([n][m])p)[0:n][1]) if(n > 2) device(dev) nowait
#pragma omp task depend(in : (([n][m])p)[0:n][1]) affinity((([n][m])p)[0:n][1])
  ;
}

// Fewer sections than dimensions: the descriptor element is a row of k doubles.
// CHECK-LABEL: define {{.*}}void @partial_depth(
// CHECK:       store i32 %k, ptr [[K_ADDR:%.+]],
void partial_depth(double *p, int n, int m, int k) {
  // CHECK:      [[K:%.+]] = load i32, ptr [[K_ADDR]],
  // CHECK-NEXT: [[K64:%.+]] = zext i32 [[K]] to i64
  // The entry size and the descriptor each compute the row size.
  // CHECK:      mul nuw i64 [[K64]], 8
  // CHECK:      [[ROW:%.+]] = mul nuw i64 [[K64]], 8
  // CHECK:      store i64 [[ROW]], ptr
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr @{{.+}}, ptr @{{.+}}, ptr null, ptr null)
#pragma omp target update from((([n][m][k])p)[0:n][1])
}

// Rows in depend and affinity: the size extends over the whole last row.
// CHECK-LABEL: define {{.*}}void @row_sections(
void row_sections(double *p, int n, int m) {
  // CHECK:      [[RS_ROW:%.+]] = mul nuw i64 %{{.+}}, 8
  // CHECK-NEXT: [[RS_END:%.+]] = getelementptr i8, ptr %{{.+}}, i64 [[RS_ROW]]
  // CHECK-NEXT: [[RS_END_INT:%.+]] = ptrtoaddr ptr [[RS_END]] to i64
  // CHECK-NEXT: [[RS_BEGIN_INT:%.+]] = ptrtoaddr ptr %{{.+}} to i64
  // CHECK-NEXT: [[RS_SIZE:%.+]] = sub nuw i64 [[RS_END_INT]], [[RS_BEGIN_INT]]
  // CHECK:      store i64 [[RS_SIZE]], ptr
#pragma omp task depend(in : (([n][m])p)[0:n]) affinity((([n][m])p)[0:n])
  ;
}

// The base pointer is the shifted pointer, not 'p'.
// CHECK-LABEL: define {{.*}}void @offset_base(
void offset_base(double *p) {
  // CHECK:      [[OB_BASE:%.+]] = getelementptr inbounds{{.*}} double, ptr %{{.+}}, i64 8
  // CHECK:      store ptr [[OB_BASE]], ptr
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr @{{.+}}, ptr @{{.+}}, ptr null, ptr null)
#pragma omp target update to((([2][4])(p + 8))[0:2][1])
}

struct T {
  int i;
  float x;
  double *p;
};

// The shaped section is not part of the combined entry of the struct.
// CHECK-LABEL: define {{.*}}void @member_and_shaped(
void member_and_shaped(struct T t, int n, int m) {
  // CHECK: call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 4, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, ptr [[MIXED_TYPES]], ptr null, ptr null)
#pragma omp target update to(t.i, t.x, (([n][m])t.p)[0:n][1])
}

// A shaped array of pointers: the base pointer is 'pp', not a section of it.
// CHECK-LABEL: define {{.*}}void @pointer_array(
// CHECK:       store ptr %pp, ptr [[PA_PP_ADDR:%.+]],
void pointer_array(double **pp, int n) {
  // CHECK:      [[PA_BP:%.+]] = load ptr, ptr [[PA_PP_ADDR]],
  // CHECK:      store ptr [[PA_BP]], ptr
  // CHECK:      getelementptr inbounds [3 x %struct.descriptor_dim{{.*}}], ptr %{{.+}}, i64 0, i64 0
  // CHECK:      call void @__tgt_target_data_update_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr %{{.+}}, ptr %{{.+}}, ptr @{{.+}}, ptr @{{.+}}, ptr null, ptr null)
#pragma omp target update to((([n])pp)[0:n][1])
}

#endif // HEADER
