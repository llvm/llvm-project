// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Inline constructors and destructors are missing comdat group declarations.
// This is a divergence from CodeGen's behavior.
//
// Per the Itanium C++ ABI:
// - Inline functions with linkonce_odr linkage should be in comdat groups
// - This allows the linker to merge duplicate definitions
// - Prevents ODR violations across translation units
//
// CodeGen correctly generates comdat declarations:
//   $_ZN1SC1Ev = comdat any
//   $_ZN1SD1Ev = comdat any
//   define linkonce_odr void @_ZN1SC1Ev(...) comdat
//
// CIR omits the comdat declarations and attribute:
//   define linkonce_odr void @_ZN1SC1Ev(...)  // No comdat!
//
// This affects:
// - All inline constructors (C1 and C2 variants)
// - All inline destructors (D1 and D2 variants)
// - Implicitly-defined constructors/destructors
// - Defaulted constructors/destructors
// - Delegating constructors
//
// Impact:
// - May cause ODR violations with multiple translation units
// - Linker cannot merge duplicate definitions
// - Potential code bloat from duplicate definitions

// DIFF: -$_ZN1SC1Ev = comdat any
// DIFF: -$_ZN1SC2Ev = comdat any
// DIFF: -$_ZN1SD1Ev = comdat any
// DIFF: -$_ZN1SD2Ev = comdat any
// DIFF: -define linkonce_odr {{.*}} @_ZN1SC1Ev{{.*}} comdat
// DIFF: +define linkonce_odr {{.*}} @_ZN1SC1Ev

struct S {
    int x;

    // Inline constructor
    S() : x(42) {}

    // Inline destructor
    ~S() {}
};

int test_ctor_dtor() {
    S s;
    return s.x;
}

// Test with parameterized constructor
struct WithParams {
    int a, b;

    // DIFF: -$_ZN10WithParamsC1Eii = comdat any
    // DIFF: +# Missing comdat declaration

    WithParams(int x, int y) : a(x), b(y) {}
    ~WithParams() {}
};

int test_params() {
    WithParams w(10, 20);
    return w.a + w.b;
}

// Test with delegating constructor
struct Delegating {
    int x, y;

    // DIFF: -$_ZN10DelegatingC1Ei = comdat any
    // DIFF: -$_ZN10DelegatingC1Eii = comdat any

    Delegating(int a) : Delegating(a, a * 2) {}
    Delegating(int a, int b) : x(a), y(b) {}
    ~Delegating() {}
};

int test_delegating() {
    Delegating d(5);
    return d.x + d.y;
}

// Test with copy constructor
struct WithCopy {
    int val;

    // DIFF: -$_ZN8WithCopyC1Ei = comdat any
    // DIFF: -$_ZN8WithCopyC1ERKS_ = comdat any

    WithCopy(int v) : val(v) {}
    WithCopy(const WithCopy& other) : val(other.val * 2) {}
    ~WithCopy() {}
};

int test_copy() {
    WithCopy w1(10);
    WithCopy w2(w1);
    return w2.val;
}

// Test with move constructor
struct WithMove {
    int* ptr;

    // DIFF: -$_ZN8WithMoveC1Ei = comdat any
    // DIFF: -$_ZN8WithMoveC1EOS_ = comdat any

    WithMove(int v) : ptr(new int(v)) {}
    WithMove(WithMove&& other) : ptr(other.ptr) { other.ptr = nullptr; }
    ~WithMove() { delete ptr; }

    int get() const { return ptr ? *ptr : 0; }
};

int test_move() {
    WithMove w1(42);
    WithMove w2(static_cast<WithMove&&>(w1));
    return w2.get();
}
