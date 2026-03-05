// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR crashes when handling static local variables with constructors.
//
// Static locals with non-trivial constructors require thread-safe initialization
// using guard variables and the __cxa_guard_acquire/__cxa_guard_release ABI.
//
// Per the Itanium C++ ABI:
// - A guard variable tracks initialization state
// - __cxa_guard_acquire checks if already initialized (returns 0 if so)
// - Constructor runs once
// - __cxa_guard_release marks as initialized
//
// Currently, CIR crashes with:
//   NYI: thread-safe guards with __cxa_guard_acquire/release
//   UNREACHABLE executed at LoweringPrepare.cpp:938
//   at lowerGuardedInitOp
//
// This affects any function with static local variables that have constructors.

struct GlobalClass {
    int value;
    GlobalClass(int v) : value(v) {}
    ~GlobalClass() {}
};

// Static local with constructor
int get_static_local() {
    static GlobalClass local(123);
    return local.value;
}

// LLVM: Should have function definition
// LLVM: define {{.*}} @_Z16get_static_localv()

// OGCG: Should use guard variables and cxa_guard functions
// OGCG: define {{.*}} @_Z16get_static_localv()
// OGCG: call {{.*}} @__cxa_guard_acquire
// OGCG: call {{.*}} @__cxa_guard_release
