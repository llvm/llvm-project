// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclang-abi-compat=23 -emit-llvm -o - %s | FileCheck %s --check-prefix=LEGACY

typedef _Float16 v1hf __attribute__((vector_size(2)));
typedef _Float16 v2hf __attribute__((vector_size(4)));
typedef _Float16 v4hf __attribute__((vector_size(8)));
typedef float v1sf __attribute__((vector_size(4)));
typedef float v2sf __attribute__((vector_size(8)));
typedef double v1df __attribute__((vector_size(8)));
typedef int v1si __attribute__((vector_size(4)));

// Single-element floating-point vectors are passed in memory for GCC
// compatibility.
// CHECK-LABEL: define{{.*}} half @take_v1hf(ptr noundef byval(<1 x half>) align 8
// LEGACY-LABEL: define{{.*}} half @take_v1hf(i16 noundef
_Float16 take_v1hf(v1hf x) { return x[0]; }

// CHECK-LABEL: define{{.*}} float @take_v1sf(ptr noundef byval(<1 x float>) align 8
// LEGACY-LABEL: define{{.*}} float @take_v1sf(i32 noundef
float take_v1sf(v1sf x) { return x[0]; }

// CHECK-LABEL: define{{.*}} double @take_v1df(ptr noundef byval(<1 x double>) align 8
// LEGACY-LABEL: define{{.*}} double @take_v1df(ptr noundef byval(<1 x double>) align 8
double take_v1df(v1df x) { return x[0]; }

// Single-element floating-point vector returns use memory too.
// CHECK-LABEL: define{{.*}} void @return_v1hf(
// CHECK-SAME: ptr {{.*}}sret(<1 x half>) align 2 %agg.result,
// CHECK-SAME: ptr noundef byval(<1 x half>) align 8
// LEGACY-LABEL: define{{.*}} i16 @return_v1hf(i16 noundef
v1hf return_v1hf(v1hf x) { return x; }

// CHECK-LABEL: define{{.*}} void @return_v1sf(
// CHECK-SAME: ptr {{.*}}sret(<1 x float>) align 4 %agg.result,
// CHECK-SAME: ptr noundef byval(<1 x float>) align 8
// LEGACY-LABEL: define{{.*}} i32 @return_v1sf(i32 noundef
v1sf return_v1sf(v1sf x) { return x; }

// CHECK-LABEL: define{{.*}} void @return_v1df(
// CHECK-SAME: ptr {{.*}}sret(<1 x double>) align 8 %agg.result,
// CHECK-SAME: ptr noundef byval(<1 x double>) align 8
// LEGACY-LABEL: define{{.*}} <1 x double> @return_v1df(
// LEGACY-SAME: ptr noundef byval(<1 x double>) align 8
v1df return_v1df(v1df x) { return x; }

// Multi-element floating-point vectors are classified as SSE.
// CHECK-LABEL: define{{.*}} half @take_v2hf(<2 x half> noundef
_Float16 take_v2hf(v2hf x) { return x[0]; }

// CHECK-LABEL: define{{.*}} half @take_v4hf(<4 x half> noundef
_Float16 take_v4hf(v4hf x) { return x[0]; }

// CHECK-LABEL: define{{.*}} float @take_v2sf(double noundef
float take_v2sf(v2sf x) { return x[0]; }

// Small integer vectors stay INTEGER.
// CHECK-LABEL: define{{.*}} i32 @take_v1si(i32 noundef
int take_v1si(v1si x) { return x[0]; }
