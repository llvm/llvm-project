// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR does not generate several important LLVM attributes on functions and parameters.
//
// Missing attributes affect:
// 1. Optimization opportunities (noundef, nonnull, dereferenceable allow more aggressive opts)
// 2. Undefined behavior detection (noundef makes undef/poison values explicit violations)
// 3. Memory safety analysis (nonnull, dereferenceable help catch null pointer bugs)
// 4. Link-time optimization (unnamed_addr allows more merging)
//
// Current divergences:
//
// Parameter attributes:
// - noundef: Parameter must not be undef or poison (helps catch UB)
// - nonnull: Pointer parameter must not be null
// - dereferenceable(N): Pointer must be dereferenceable for at least N bytes
//
// Function attributes:
// - mustprogress: Function must make forward progress (no infinite loops without side effects)
// - unnamed_addr: Function address is not semantically significant
//
// Missing metadata:
// - Function attributes: min-legal-vector-width, target-features, stack-protector-buffer-size
//
// Impact: Medium - Reduces optimization quality and UB detection

struct S {
    int x;
    S(int v) : x(v) {}
    ~S() {}
};

// Test parameter attributes
int process_struct(S* s, int value) {
    return s->x + value;
}

// Test return and 'this' pointer attributes
int S_get_value(S* s) {
    return s->x;
}

// Test reference parameters
void take_reference(const S& s) {
}

// LLVM: Missing noundef, nonnull, dereferenceable
// LLVM: define {{.*}} i32 @_Z14process_structP1Si(ptr %0, i32 %1)
// LLVM-NOT: noundef
// LLVM-NOT: nonnull
// LLVM-NOT: dereferenceable

// OGCG: Should have all attributes
// OGCG: define {{.*}} noundef i32 @_Z14process_structP1Si(ptr noundef %s, i32 noundef %value)
// OGCG: define {{.*}} noundef i32 @_Z12S_get_valueP1S(ptr noundef %s)
// OGCG: define {{.*}} void @_Z14take_referenceRK1S(ptr noundef nonnull align 4 dereferenceable(4) %s)
