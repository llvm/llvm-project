// RUN: %clang_cc1 -verify -fsyntax-only %s

// C11 6.2.2p7: same identifier with both internal and external linkage is UB.

// Conflicting linkage (UB)

static int x; // expected-note {{previous}}
void test_basic_shadow(void) {
    int x;
    { extern int x; } // expected-error {{declared with both internal and external linkage}}
}

static int y; // expected-note {{previous}}
void test_deep_nesting(void) {
    int y;
    { int y; { { extern int y; } } } // expected-error {{declared with both internal and external linkage}}
}

static int p; // expected-note {{previous}}
void test_param_shadow(int p) {
    { extern int p; } // expected-error {{declared with both internal and external linkage}}
}

// Valid cases

static int a;
void test_no_shadow(void) {
    extern int a;
}

void test_no_file_scope(void) {
    for (static int b = 0;;) {
        extern int b;
        break;
    }
}
