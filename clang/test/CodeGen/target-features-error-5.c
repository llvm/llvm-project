// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -target-feature +sse4.1 -emit-llvm -verify -o /dev/null

typedef double __v2df __attribute__((__vector_size__(16)));

__v2df foo() {
    __v2df v = {0.0, 0.0};
    return __builtin_ia32_roundpd(v, 2);
}

__v2df __attribute__((target("no-sse4.1"), flatten)) bar() {
    return foo(); // expected-error {{flatten function 'bar' calls 'foo' which requires target feature 'sse4.1', but the caller is compiled without support for 'sse4.1'}}
}
