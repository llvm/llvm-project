// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fcxx-exceptions -fexceptions %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fcxx-exceptions -fexceptions %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR crashes when handling C++ exceptions (try/catch blocks).
//
// Exception handling requires:
// - Generating personality functions
// - Landing pads for catch blocks
// - Invoke instructions instead of call for functions that may throw
// - Exception object allocation and cleanup
//
// Currently, CIR crashes with:
//   NYI
//   UNREACHABLE executed at CIRGenItaniumCXXABI.cpp:814
//   at emitBeginCatch
//
// This affects any code using try/catch/throw.

struct Exception {
    int code;
    Exception(int c) : code(c) {}
    ~Exception() {}
};

void may_throw() {
    throw Exception(42);
}

int catch_exception() {
    try {
        may_throw();
        return 0;
    } catch (const Exception& e) {
        return e.code;
    }
}

// LLVM: Should generate exception handling code
// LLVM: define {{.*}} @_Z15catch_exceptionv()

// OGCG: Should use invoke and landing pads
// OGCG: define {{.*}} @_Z15catch_exceptionv() {{.*}} personality
// OGCG: invoke {{.*}} @_Z9may_throwv()
// OGCG: landingpad
