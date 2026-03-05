// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR generates template instantiations without comdat groups.
//
// Template instantiations should be marked with comdat groups to ensure
// proper ODR (One Definition Rule) compliance when linking multiple TUs
// that instantiate the same template.
//
// Current divergences:
// 1. CIR: define linkonce_odr void @_ZN7WrapperIiEC1Ei(...)
//    CodeGen: define linkonce_odr void @_ZN7WrapperIiEC1Ei(...) comdat
//
// 2. CIR: define linkonce_odr i32 @_Z3addIiET_S0_S0_(...)
//    CodeGen: define linkonce_odr i32 @_Z3addIiET_S0_S0_(...) comdat
//
// Without comdat, the linker may fail to properly merge duplicate definitions,
// leading to ODR violations, increased binary size, or linker errors.

template<typename T>
struct Wrapper {
    T value;
    Wrapper(T v) : value(v) {}
    T get() const { return value; }
};

template<typename T>
T add(T a, T b) {
    return a + b;
}

int test_templates() {
    Wrapper<int> w(42);
    return w.get() + add(1, 2);
}

// LLVM: Template instantiations exist
// LLVM: define linkonce_odr {{.*}} @_ZN7WrapperIiEC1Ei
// LLVM: define linkonce_odr {{.*}} @_Z3addIiET_S0_S0_

// OGCG: Template instantiations should have comdat
// OGCG: define linkonce_odr {{.*}} @_ZN7WrapperIiEC1Ei({{.*}}) {{.*}} comdat
// OGCG: define linkonce_odr {{.*}} @_Z3addIiET_S0_S0_({{.*}}) {{.*}} comdat
