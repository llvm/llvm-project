// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

template <class T> struct K {
  T v;
  virtual ~K() { v = T(); }
};
template class K<double>;

// CIR: cir.func {{.*}} comdat("_ZN1KIdED5Ev") {{.*}} weak_odr @_ZN1KIdED2Ev(
// CIR: cir.func weak_odr private @_ZN1KIdED1Ev(!cir.ptr<!rec_K3Cdouble3E>) alias(@_ZN1KIdED2Ev)

// LLVM: $_ZN1KIdED5Ev = comdat any
// LLVM: @_ZN1KIdED1Ev = weak_odr {{.*}}alias void (ptr), ptr @_ZN1KIdED2Ev
// LLVM: define weak_odr void @_ZN1KIdED2Ev({{.*}}) {{.*}} comdat($_ZN1KIdED5Ev)
