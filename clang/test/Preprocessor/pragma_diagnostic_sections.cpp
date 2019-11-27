// RUN: %clang_cc1 -fsyntax-only -Wall -Wunused-macros -Wunused-parameter -Wno-uninitialized -Wno-misleading-indentation -verify %s

// rdar://8365684
struct S {
    void m1() { int b; while (b==b); } // expected-warning {{always evaluates to true}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"
    void m2() { int b; while (b==b); }
#pragma clang diagnostic pop

    void m3() { int b; while (b==b); } // expected-warning {{always evaluates to true}}
};

//------------------------------------------------------------------------------

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"
template <typename T>
struct TS {
    void m() { T b; while (b==b); }
};
#pragma clang diagnostic pop

void f() {
    TS<int> ts;
    ts.m();
}

//------------------------------------------------------------------------------

#define UNUSED_MACRO1 // expected-warning {{macro is not used}}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-macros"
#define UNUSED_MACRO2
#pragma clang diagnostic pop

//------------------------------------------------------------------------------

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
int g() { }
#pragma clang diagnostic pop

//------------------------------------------------------------------------------

void ww(
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
        int x,
#pragma clang diagnostic pop
        int y) // expected-warning {{unused}}
{
}

//------------------------------------------------------------------------------

struct S2 {
    int x, y;
    S2() : 
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreorder"
    y(),
    x()
#pragma clang diagnostic pop
    {}
};

//------------------------------------------------------------------------------

// rdar://8790245
#define MYMACRO \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wunknown-pragmas\"") \
    _Pragma("clang diagnostic pop")
MYMACRO
#undef MYMACRO

//------------------------------------------------------------------------------
