// RUN: %clang_cc1 -verify -fsyntax-only -std=c++17 %s

// expected-no-diagnostics

// In C++, block-scope extern declarations target the enclosing namespace
// scope ([dcl.meaning.general]/3.5), so they find the file-scope static
// despite local shadows and inherit internal linkage. No conflict arises.
//
// This differs from C, where a local shadow breaks linkage inheritance,
// causing the conflict diagnosed by err_internal_extern_mismatch.

// Example adapted from [basic.link]/6.
static void f();
static int i = 0;
void g() {
    extern void f();   // internal linkage
    int i;             // no linkage
    {
        extern void f(); // internal linkage
        extern int i;    // internal linkage
    }
}
