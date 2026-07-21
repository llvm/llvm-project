// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=SYSV
// RUN: %clang_cc1 -triple x86_64-scei-ps4 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=PS
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=PS

typedef float v8f __attribute__((vector_size(32)));
typedef float v16f __attribute__((vector_size(64)));

v8f g256(v8f x) { return x; }
v16f g512(v16f x) { return x; }

__attribute__((target("avx"))) v8f l256(v8f x) { return x; }
__attribute__((target("avx512f"))) v16f l512(v16f x) { return x; }

__attribute__((target("avx"))) v8f call_l256(v8f x) { return l256(x); }
__attribute__((target("avx512f"))) v16f call_l512(v16f x) { return l512(x); }

__attribute__((target("avx"))) v8f call_ptr_l256(v8f x) {
  v8f (*fp)(v8f) = l256;
  return fp(x);
}

__attribute__((target("avx512f"))) v16f call_ptr_l512(v16f x) {
  v16f (*fp)(v16f) = l512;
  return fp(x);
}

// SYSV-LABEL: define dso_local <8 x float> @g256(
// SYSV: byval(<8 x float>) align 32

// SYSV-LABEL: define dso_local <16 x float> @g512(
// SYSV: byval(<16 x float>) align 64

// SYSV-LABEL: define dso_local <8 x float> @l256(<8 x float> noundef %x)
// SYSV-LABEL: define dso_local <16 x float> @l512(<16 x float> noundef %x)

// SYSV-LABEL: define dso_local <8 x float> @call_l256(<8 x float> noundef %x)
// SYSV: call <8 x float> @l256(<8 x float> noundef

// SYSV-LABEL: define dso_local <16 x float> @call_l512(<16 x float> noundef %x)
// SYSV: call <16 x float> @l512(<16 x float> noundef

// SYSV-LABEL: define dso_local <8 x float> @call_ptr_l256(<8 x float> noundef %x)
// SYSV: call <8 x float> %{{.*}}(<8 x float> noundef

// SYSV-LABEL: define dso_local <16 x float> @call_ptr_l512(<16 x float> noundef %x)
// SYSV: call <16 x float> %{{.*}}(<16 x float> noundef

// PlayStation keeps the legacy ABI which always returns AVX vectors in registers & only uses AVX level from module/TU level even with AVX target attributes.
// PS-LABEL: define dso_local <8 x float> @g256(
// PS: byval(<8 x float>) align 32

// PS-LABEL: define dso_local <16 x float> @g512(
// PS: byval(<16 x float>) align 64

// PS-LABEL: define dso_local <8 x float> @l256(
// PS: byval(<8 x float>) align 32

// PS-LABEL: define dso_local <16 x float> @l512(
// PS: byval(<16 x float>) align 64

// PS-LABEL: define dso_local <8 x float> @call_l256(
// PS: call <8 x float> @l256(ptr noundef byval(<8 x float>) align 32

// PS-LABEL: define dso_local <16 x float> @call_l512(
// PS: call <16 x float> @l512(ptr noundef byval(<16 x float>) align 64

// PS-LABEL: define dso_local <8 x float> @call_ptr_l256(
// PS: call <8 x float> %{{.*}}(ptr noundef byval(<8 x float>) align 32

// PS-LABEL: define dso_local <16 x float> @call_ptr_l512(
// PS: call <16 x float> %{{.*}}(ptr noundef byval(<16 x float>) align 64
