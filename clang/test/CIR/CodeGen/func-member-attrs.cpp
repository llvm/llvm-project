// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s -check-prefix=CIR --input-file %t.cir
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file %t-cir.ll
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=LLVM --input-file %t.ll

struct S {
  void regular_method();

  void inline_method() {}

  void __attribute__((aligned(8))) aligned_method() {}

  static void static_method() {}
};

void S::regular_method() {}

inline void free_inline() {}

void use(S &s) {
  s.regular_method();
  s.inline_method();
  s.aligned_method();
  S::static_method();
  free_inline();
}

// CIR: cir.func no_inline alignment(2) dso_local @_ZN1S14regular_methodEv(
// LLVM: define dso_local void @_ZN1S14regular_methodEv({{.*}}) #{{[0-9]+}} align 2

// CIR: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN1S13inline_methodEv(
// LLVM: define linkonce_odr void @_ZN1S13inline_methodEv({{.*}}) #{{[0-9]+}} comdat align 2

// CIR: cir.func no_inline comdat alignment(8) linkonce_odr @_ZN1S14aligned_methodEv(
// LLVM: define linkonce_odr void @_ZN1S14aligned_methodEv({{.*}}) #{{[0-9]+}} comdat align 8

// CIR: cir.func no_inline comdat alignment(2) linkonce_odr @_ZN1S13static_methodEv(
// LLVM: define linkonce_odr void @_ZN1S13static_methodEv() #{{[0-9]+}} comdat align 2

// CIR: cir.func no_inline comdat linkonce_odr @_Z11free_inlinev()
// LLVM: define linkonce_odr void @_Z11free_inlinev() #{{[0-9]+}} comdat {
