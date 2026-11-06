// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -fclangir -verify -o /dev/null

typedef double __v2df __attribute__((__vector_size__(16)));

__v2df __attribute__((target("sse4.1"))) foo() {
    __v2df v = {0.0, 0.0};
    return v;
}

__v2df __attribute__((target("sse4.1"), flatten)) bar() {
    return foo(); // expected-no-diagnostics
}
