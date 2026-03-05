// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR does not generate thread wrapper functions for thread-local variables.
//
// The Itanium C++ ABI requires thread wrapper functions for non-local
// thread-local variables to properly initialize them on first access.
// The wrapper is named __tls_wrapper_<mangled_name> or _ZTW<mangled_name>.
//
// Current divergence:
// CIR: Does not generate @_ZTW7tls_var wrapper function
// CodeGen: Generates weak_odr hidden ptr @_ZTW7tls_var() comdat
//
// Without the wrapper, thread-local initialization may not work correctly
// when the variable is accessed from other translation units.

thread_local int tls_var = 42;

int get_tls() {
    return tls_var;
}

void set_tls(int val) {
    tls_var = val;
}

// LLVM: Should define functions
// LLVM: define {{.*}} @_Z7get_tlsv()
// LLVM: define {{.*}} @_Z7set_tlsi({{.*}})

// OGCG: Should generate thread wrapper function
// OGCG: define weak_odr hidden {{.*}} ptr @_ZTW7tls_var() {{.*}} comdat
// OGCG: $_ZTW7tls_var = comdat any
