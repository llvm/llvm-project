// RUN: %clang_cc1 -verify -fsyntax-only -std=c++17 %s

// expected-no-diagnostics

// CWG 426 / [basic.link]p6: the same identifier with both internal and
// external linkage should be ill-formed in C++, but Clang does not yet
// diagnose this due to ABI break concerns. This test documents current
// behavior.

static int x;
void test_shadow(void) {
    int x;
    {
        // FIXME: Per CWG 426, this should be ill-formed because the
        // file-scope 'x' has internal linkage but this 'extern' gets
        // external linkage due to the local shadow.
        extern int x;
    }
}
