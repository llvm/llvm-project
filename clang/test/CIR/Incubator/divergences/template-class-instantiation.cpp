// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Template class instantiation missing comdat.
//
// CodeGen:
//   $_ZN9ContainerIiEC1Ei = comdat any
//   define linkonce_odr void @_ZN9ContainerIiEC1Ei(...) comdat
//
// CIR:
//   define linkonce_odr void @_ZN9ContainerIiEC1Ei(...)  // No comdat

// DIFF: -$_ZN9ContainerIiEC1Ei = comdat any
// DIFF: -define linkonce_odr {{.*}} comdat
// DIFF: +define linkonce_odr

template<typename T>
struct Container {
    T value;
    Container(T v) : value(v) {}
    T get() const { return value; }
};

int test() {
    Container<int> c(42);
    return c.get();
}
