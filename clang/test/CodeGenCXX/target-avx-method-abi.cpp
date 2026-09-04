// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=SYSV
// RUN: %clang_cc1 -triple x86_64-scei-ps4 -std=c++20 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=PS
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -std=c++20 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=PS

typedef float v8f __attribute__((vector_size(32)));
typedef float v16f __attribute__((vector_size(64)));

struct S {
  __attribute__((target("avx"))) 
  v8f m(v8f x) { return x; }
};

struct C {
  v8f field;
  __attribute__((target("avx"))) 
  C(v8f x) : field(x) {}
};

__attribute__((target("avx"))) 
v8f callm(S *s, v8f x) { 
  return s->m(x); 
}

__attribute__((target("avx"))) 
v8f callctor(v8f x) {
  C c(x);
  return c.field;
}

__attribute__((target("avx")))
v8f callm_ptr(S *s, v8f x) {
  v8f (S::*pmf)(v8f) = &S::m;
  return (s->*pmf)(x);
}

struct S512 {
  __attribute__((target("avx512f")))
  v16f m512(v16f x) { return x; }
};

struct D512 {
  v16f field;
  __attribute__((target("avx512f")))
  D512(v16f x) : field(x) {}
};

__attribute__((target("avx512f")))
v16f callm512(S512 *s, v16f x) {
  return s->m512(x);
}

__attribute__((target("avx512f")))
v16f callctor512(v16f x) {
  D512 c(x);
  return c.field;
}

__attribute__((target("avx512f")))
v16f callm512_ptr(S512 *s, v16f x) {
  v16f (S512::*pmf)(v16f) = &S512::m512;
  return (s->*pmf)(x);
}

// Desired ABI behavior: AVX-targeted member functions should pass/return AVX
// vectors directly, just like AVX-targeted free functions.
// SYSV-LABEL: define dso_local noundef <8 x float> @_Z5callmP1SDv8_f(
// SYSV: call noundef <8 x float> @_ZN1S1mEDv8_f(ptr noundef
// SYSV-LABEL: define linkonce_odr noundef <8 x float> @_ZN1S1mEDv8_f(
// SYSV-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this, <8 x float> noundef %x)
// SYSV-LABEL: define dso_local noundef <8 x float> @_Z8callctorDv8_f(
// SYSV: call void @_ZN1CC1EDv8_f(ptr noundef nonnull align 32 dereferenceable(32)
// SYSV-SAME: <8 x float> noundef
// SYSV-LABEL: define linkonce_odr void @_ZN1CC1EDv8_f(
// SYSV-SAME: ptr noundef nonnull align 32 dereferenceable(32) %this, <8 x float> noundef %x)
// SYSV-LABEL: define dso_local noundef <8 x float> @_Z9callm_ptrP1SDv8_f(
// SYSV-SAME: ptr noundef %s, <8 x float> noundef %x)
// SYSV: call noundef <8 x float> %{{.*}}(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}}, <8 x float> noundef
// SYSV-LABEL: define dso_local noundef <16 x float> @_Z8callm512P4S512Dv16_f(
// SYSV: call noundef <16 x float> @_ZN4S5124m512EDv16_f(ptr noundef
// SYSV-LABEL: define linkonce_odr noundef <16 x float> @_ZN4S5124m512EDv16_f(
// SYSV-SAME: ptr noundef nonnull align 1 dereferenceable(1) %this, <16 x float> noundef %x)
// SYSV-LABEL: define dso_local noundef <16 x float> @_Z11callctor512Dv16_f(
// SYSV: call void @_ZN4D512C1EDv16_f(ptr noundef nonnull align 64 dereferenceable(64)
// SYSV-SAME: <16 x float> noundef
// SYSV-LABEL: define linkonce_odr void @_ZN4D512C1EDv16_f(
// SYSV-SAME: ptr noundef nonnull align 64 dereferenceable(64) %this, <16 x float> noundef %x)
// SYSV-LABEL: define dso_local noundef <16 x float> @_Z12callm512_ptrP4S512Dv16_f(
// SYSV-SAME: ptr noundef %s, <16 x float> noundef %x)
// SYSV: call noundef <16 x float> %{{.*}}(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}}, <16 x float> noundef
// SYSV-LABEL: define linkonce_odr void @_ZN1CC2EDv8_f(
// SYSV-SAME: ptr noundef nonnull align 32 dereferenceable(32) %this, <8 x float> noundef %x)
// SYSV-LABEL: define linkonce_odr void @_ZN4D512C2EDv16_f(
// SYSV-SAME: ptr noundef nonnull align 64 dereferenceable(64) %this, <16 x float> noundef %x)

// PlayStation keeps the legacy ABI which always returns AVX vectors in registers & only uses AVX level from module/TU level even with AVX target attributes.
// PS-LABEL: define dso_local noundef <8 x float> @_Z5callmP1SDv8_f(
// PS: byval(<8 x float>) align 32
// PS: call noundef <8 x float> @_ZN1S1mEDv8_f(
// PS-LABEL: define linkonce_odr noundef <8 x float> @_ZN1S1mEDv8_f(
// PS: byval(<8 x float>) align 32
// PS-LABEL: define dso_local noundef <8 x float> @_Z8callctorDv8_f(
// PS-LABEL: define linkonce_odr void @_ZN1CC1EDv8_f(
// PS-SAME: ptr noundef nonnull align 32 dereferenceable(32) %this, ptr noundef byval(<8 x float>) align 32
// PS-LABEL: define dso_local noundef <8 x float> @_Z9callm_ptrP1SDv8_f(
// PS-SAME: ptr noundef %s, ptr noundef byval(<8 x float>) align 32
// PS: call noundef <8 x float> %{{.*}}(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}}, ptr noundef byval(<8 x float>) align 32
// PS-LABEL: define dso_local noundef <16 x float> @_Z8callm512P4S512Dv16_f(
// PS: byval(<16 x float>) align 64
// PS: call noundef <16 x float> @_ZN4S5124m512EDv16_f(
// PS-LABEL: define linkonce_odr noundef <16 x float> @_ZN4S5124m512EDv16_f(
// PS: byval(<16 x float>) align 64
// PS-LABEL: define dso_local noundef <16 x float> @_Z11callctor512Dv16_f(
// PS-LABEL: define linkonce_odr void @_ZN4D512C1EDv16_f(
// PS-SAME: ptr noundef nonnull align 64 dereferenceable(64) %this, ptr noundef byval(<16 x float>) align 64
// PS-LABEL: define dso_local noundef <16 x float> @_Z12callm512_ptrP4S512Dv16_f(
// PS-SAME: ptr noundef %s, ptr noundef byval(<16 x float>) align 64
// PS: call noundef <16 x float> %{{.*}}(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}}, ptr noundef byval(<16 x float>) align 64
// PS-LABEL: define linkonce_odr void @_ZN1CC2EDv8_f(
// PS-SAME: ptr noundef nonnull align 32 dereferenceable(32) %this, ptr noundef byval(<8 x float>) align 32
// PS-LABEL: define linkonce_odr void @_ZN4D512C2EDv16_f(
// PS-SAME: ptr noundef nonnull align 64 dereferenceable(64) %this, ptr noundef byval(<16 x float>) align 64
