// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-feature +avx %s -emit-llvm -o - | FileCheck %s --check-prefix=AVX
// RUN: %clang_cc1 -triple x86_64-linux-gnu -target-feature +avx -fclang-abi-compat=23 %s -emit-llvm -o - | FileCheck %s --check-prefix=LEGACY

struct F3 {
  float a[3];
};

typedef _Atomic(float) atomic_float;
typedef _Atomic(struct F3) atomic_f3;
typedef _Float16 v1hf __attribute__((vector_size(2)));
typedef __bf16 v1bf __attribute__((vector_size(2)));
typedef float v1sf __attribute__((vector_size(4)));
typedef double v1df __attribute__((vector_size(8)));
typedef __int128 v2ti __attribute__((vector_size(32)));
typedef _Atomic(v1hf) atomic_v1hf;
typedef _Atomic(v1bf) atomic_v1bf;
typedef _Atomic(v1sf) atomic_v1sf;
typedef _Atomic(v1df) atomic_v1df;
typedef _Atomic(v2ti) atomic_v2ti;

// CHECK-LABEL: define dso_local void @take_float(
// CHECK-SAME: float %x)
void take_float(atomic_float x) {}

// CHECK-LABEL: define dso_local float @ret_float(
// CHECK-SAME: float %x)
atomic_float ret_float(atomic_float x) {
  return x;
}

// CHECK-LABEL: define dso_local void @take_f3(
// CHECK-SAME: ptr noundef byval({ %struct.F3, [4 x i8] }) align 16 %x)
void take_f3(atomic_f3 x) {}

// CHECK-LABEL: define dso_local void @ret_f3(
// CHECK-SAME: ptr {{.*}}sret({ %struct.F3, [4 x i8] }) align 16 %agg.result,
// CHECK-SAME: ptr noundef byval({ %struct.F3, [4 x i8] }) align 16 %x)
atomic_f3 ret_f3(atomic_f3 x) {
  return x;
}

// AVX-LABEL: define dso_local void @ret_v1hf(
// AVX-SAME: ptr {{.*}}sret(<1 x half>) align 2 %agg.result,
// AVX-SAME: ptr noundef byval(<1 x half>) align 8 %{{.*}})
atomic_v1hf ret_v1hf(atomic_v1hf x) {
  return x;
}

// AVX-LABEL: define dso_local void @ret_v1bf(
// AVX-SAME: ptr {{.*}}sret(<1 x bfloat>) align 2 %agg.result,
// AVX-SAME: ptr noundef byval(<1 x bfloat>) align 8 %{{.*}})
atomic_v1bf ret_v1bf(atomic_v1bf x) {
  return x;
}

// AVX-LABEL: define dso_local void @take_v1sf(
// AVX-SAME: ptr noundef byval(<1 x float>) align 8 %{{.*}})
// LEGACY-LABEL: define dso_local void @take_v1sf(
// LEGACY-SAME: <1 x float> %x)
void take_v1sf(atomic_v1sf x) {}

// AVX-LABEL: define dso_local void @ret_v1sf(
// AVX-SAME: ptr {{.*}}sret(<1 x float>) align 4 %agg.result,
// AVX-SAME: ptr noundef byval(<1 x float>) align 8 %{{.*}})
// LEGACY-LABEL: define dso_local <1 x float> @ret_v1sf(
// LEGACY-SAME: <1 x float> %x)
atomic_v1sf ret_v1sf(atomic_v1sf x) {
  return x;
}

// AVX-LABEL: define dso_local void @ret_v1df(
// AVX-SAME: ptr {{.*}}sret(<1 x double>) align 8 %agg.result,
// AVX-SAME: ptr noundef byval(<1 x double>) align 8 %{{.*}})
atomic_v1df ret_v1df(atomic_v1df x) {
  return x;
}

// AVX-LABEL: define dso_local void @take_v2ti(
// AVX-SAME: ptr noundef byval(<2 x i128>) align 32 %{{.*}})
// LEGACY-LABEL: define dso_local void @take_v2ti(
// LEGACY-SAME: <2 x i128> %x)
void take_v2ti(atomic_v2ti x) {}

// AVX-LABEL: define dso_local void @ret_v2ti(
// AVX-SAME: ptr {{.*}}sret(<2 x i128>) align 32 %agg.result,
// AVX-SAME: ptr noundef byval(<2 x i128>) align 32 %{{.*}})
// LEGACY-LABEL: define dso_local <2 x i128> @ret_v2ti(
// LEGACY-SAME: <2 x i128> %x)
atomic_v2ti ret_v2ti(atomic_v2ti x) {
  return x;
}
