// Tests for member evidence emitted from C++ constructor initializer lists.
// Constructor initializers (': field(expr)') are CXXCtorInitializer nodes,
// not BinaryOperator assignments, so they need dedicated evidence emission.
//
// RUN: rm -f %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nullable -std=c++17 %s \
// RUN:   --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu --ssaf-tu-summary-file=%t.json
// RUN: %python %S/Inputs/decode-summary.py %t.json | FileCheck %s --check-prefix=EVIDENCE

// ===----------------------------------------------------------------------===//
// Basic: nullable parameter -> nullable member evidence
// ===----------------------------------------------------------------------===//

struct Basic {
    int *ptr;
    // Unannotated parameter — no evidence emitted (null-unspecified, not
    // explicitly _Nullable). Evidence is only emitted for provably nullable
    // or provably nonnull sources.
    Basic(int *p) : ptr(p) {}
};

// ===----------------------------------------------------------------------===//
// Explicitly _Nullable parameter -> nullable member evidence
// ===----------------------------------------------------------------------===//

struct ExplicitNullable {
    int *ptr;
    ExplicitNullable(int * _Nullable p) : ptr(p) {}
};

// ===----------------------------------------------------------------------===//
// Nonnull parameter (_Nonnull annotation) -> nonnull member evidence
// ===----------------------------------------------------------------------===//

struct NonnullParam {
    int *ptr;
    NonnullParam(int * _Nonnull p) : ptr(p) {}
};

// ===----------------------------------------------------------------------===//
// Multiple members in initializer list
// ===----------------------------------------------------------------------===//

struct Multi {
    int *a;
    int *b;
    int *c;
    Multi(int *x, int * _Nonnull y, int *z)
        : a(x),  // unannotated — no evidence
          b(y),
          c(z) {} // unannotated — no evidence
};

// ===----------------------------------------------------------------------===//
// Non-pointer members are skipped (no evidence for int, etc.)
// ===----------------------------------------------------------------------===//

struct NonPointer {
    int val;
    int *ptr;
    NonPointer(int v, int *p)
        : val(v),
          ptr(p) {} // unannotated — no evidence
};

// ===----------------------------------------------------------------------===//
// address-of (&x) is nonnull evidence
// ===----------------------------------------------------------------------===//

struct AddrOf {
    int *ptr;
    int x;
    AddrOf() : ptr(&x) {}
};

// ===----------------------------------------------------------------------===//
// new expression is nonnull evidence
// ===----------------------------------------------------------------------===//

struct NewExpr {
    int *ptr;
    NewExpr() : ptr(new int(42)) {}
};

// ===----------------------------------------------------------------------===//
// this pointer is nonnull evidence
// ===----------------------------------------------------------------------===//

struct Base {
    int x;
};

struct UsesThis : Base {
    Base *self;
    UsesThis() : self(this) {}
};

// ===----------------------------------------------------------------------===//
// __attribute__((nonnull)) on constructor -> nonnull evidence for params
// ===----------------------------------------------------------------------===//

struct AttrNonnull {
    int *ptr;
    __attribute__((nonnull))
    AttrNonnull(int *p) : ptr(p) {}
};

// ===----------------------------------------------------------------------===//
// nullptr initializer is nullable evidence
// ===----------------------------------------------------------------------===//

struct NullInit {
    int *ptr;
    NullInit() : ptr(nullptr) {}
};

// ===----------------------------------------------------------------------===//
// Instantiation call sites (ensure no crashes, evidence still emitted)
// ===----------------------------------------------------------------------===//

void test_instantiations() {
    int x = 0;
    Basic b(&x);
    ExplicitNullable en(&x);
    NonnullParam np(&x);
    Multi m(&x, &x, &x);
    NonPointer npt(1, &x);
    AddrOf ao;
    NewExpr ne;
    UsesThis ut;
    AttrNonnull an(&x);
    NullInit ni;
}

// EVIDENCE-NOT:  {{.}}
// EVIDENCE:      c:@S@AddrOf@F@AddrOf# NonnullEvidence c:@S@AddrOf@FI@ptr
// EVIDENCE-NEXT: c:@S@AttrNonnull@F@AttrNonnull#*I# NonnullEvidence c:@S@AttrNonnull@FI@ptr
// EVIDENCE-NEXT: c:@S@Basic@F@Basic#*I# ConditionalEvidence c:@S@Basic@FI@ptr <- c:@S@Basic@F@Basic#*I# param 1
// EVIDENCE-NEXT: c:@S@ExplicitNullable@F@ExplicitNullable#*I# NullableEvidence c:@S@ExplicitNullable@FI@ptr
// EVIDENCE-NEXT: c:@S@Multi@F@Multi#*I#S0_#S0_# ConditionalEvidence c:@S@Multi@FI@a <- c:@S@Multi@F@Multi#*I#S0_#S0_# param 1
// EVIDENCE-NEXT: c:@S@Multi@F@Multi#*I#S0_#S0_# ConditionalEvidence c:@S@Multi@FI@c <- c:@S@Multi@F@Multi#*I#S0_#S0_# param 3
// EVIDENCE-NEXT: c:@S@Multi@F@Multi#*I#S0_#S0_# NonnullEvidence c:@S@Multi@FI@b
// EVIDENCE-NEXT: c:@S@NewExpr@F@NewExpr# NonnullEvidence c:@S@NewExpr@FI@ptr
// EVIDENCE-NEXT: c:@S@NonPointer@F@NonPointer#I#*I# ConditionalEvidence c:@S@NonPointer@FI@ptr <- c:@S@NonPointer@F@NonPointer#I#*I# param 2
// EVIDENCE-NEXT: c:@S@NonnullParam@F@NonnullParam#*I# NonnullEvidence c:@S@NonnullParam@FI@ptr
// EVIDENCE-NEXT: c:@S@NullInit@F@NullInit# NullableEvidence c:@S@NullInit@FI@ptr
// EVIDENCE-NEXT: c:@S@UsesThis@F@UsesThis# NonnullEvidence c:@S@UsesThis@FI@self
// EVIDENCE-NOT:  {{.}}
