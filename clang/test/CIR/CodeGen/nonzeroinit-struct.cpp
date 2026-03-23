// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - -verify

struct Other {
    int x;
};

struct Trivial {
    int x;
    double y;
    decltype(&Other::x) ptr;
};

// This case has a trivial default constructor, but can't be zero-initialized
// because it contains a data member pointer (null = -1 in Itanium ABI).
Trivial t;

// expected-error@*:* {{ClangIR code gen Not Yet Implemented: tryEmitPrivateForVarInit: non-zero-initializable cxx record}}
