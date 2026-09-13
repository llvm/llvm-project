// RUN: %clang_cc1 -Wformat-pedantic -Werror -emit-llvm -o /dev/null -verify %s

int printflike(const char *__restrict__ x, ...) __attribute__((__format__(__printf__, 1, 2)));

struct Foo {
    int *x;
};

void test() {
    __builtin_dump_struct(&(struct Foo){0}, printflike); // expected-error {{taking the address of a temporary object of type 'struct Foo'}} \
                                                         // expected-error {{format specifies type 'void *' but the argument has type 'int *'}} \
                                                         // expected-note {{in call to printing function}}
}
