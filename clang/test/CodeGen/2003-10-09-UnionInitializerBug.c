// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

struct Foo {
    unsigned a;
    unsigned b;
    unsigned c;
};

struct Bar {
    union {
        void **a;
        struct Foo b;
    }u;
};

struct Bar test = {0};

