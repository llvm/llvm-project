// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-msvc -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-msvc -o - -S %s | FileCheck -check-prefix=ASM %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-gnu -o - -emit-llvm %s | FileCheck -check-prefix=GNU-LD %s

// WinCall passes:
//   - long double as f64 (no x87),
//   - complex scalars in vector registers,
//   - a single-FP-member struct in the vector register (only when the struct
//     is exactly as big as its single member).

// long double is f64 on x86_64apx-windows-gnu.
// GNU-LD: define dso_local x86_wincallcc double @"\01f@win"(double noundef %v)
__attribute__((wincall)) long double f(long double v) { return v; }

// _Complex double travels in one XMM register.
// CHECK: define dso_local x86_wincallcc <2 x double> @"\01f_cd@win"(<2 x double> noundef %v.coerce)
// ASM-LABEL: f_cd@win:
// ASM: vmovupd %xmm0, (%rsp)
__attribute__((wincall)) _Complex double f_cd(_Complex double v) { return v; }

// _Complex float travels in one XMM register.
// CHECK: define dso_local x86_wincallcc <2 x float> @"\01f_cf@win"(<2 x float> noundef %v.coerce)
// ASM-LABEL: f_cf@win:
// ASM: vmovlpd %xmm0, (%rsp)
__attribute__((wincall)) _Complex float f_cf(_Complex float v) { return v; }

// A struct holding one double is passed like a double, in XMM0.
// CHECK: define dso_local x86_wincallcc double @"\01f_od@win"(double %s.coerce)
// ASM-LABEL: f_od@win:
__attribute__((wincall)) double f_od(struct one_double { double d; } s) {
  return s.d;
}

// A struct holding one float is passed like a float, in XMM0.
// CHECK: define dso_local x86_wincallcc float @"\01f_of@win"(float %s.coerce)
// ASM-LABEL: f_of@win:
__attribute__((wincall)) float f_of(struct one_float { float f; } s) {
  return s.f;
}

// A 16-byte-aligned struct holding one double is NOT treated as a scalar:
// its size (16) is bigger than the member (8), so it stays an aggregate and
// is expanded into its parts (the single double field; padding is dropped).
// CHECK: define dso_local x86_wincallcc double @"\01f_ad@win"(double %s.0)
__attribute__((wincall)) double f_ad(struct aligned_double { double d; } __attribute__((aligned(16))) s) {
  return s.d;
}

// A two-double struct is a normal two-register aggregate.
// CHECK: define dso_local x86_wincallcc double @"\01f_td@win"(double %s.0, double %s.1)
// ASM-LABEL: f_td@win:
__attribute__((wincall)) double f_td(struct two_double { double a, b; } s) {
  return s.a;
}
