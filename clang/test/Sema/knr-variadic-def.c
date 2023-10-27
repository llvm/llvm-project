// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s -std=c99
// PR4287

#include <stdarg.h>
char *foo = "test";
int test(char*,...);

int test(fmt) // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
        char*fmt;
{
        va_list ap;
        char*a;
        int x;

        va_start(ap,fmt);
        a=va_arg(ap,char*);
        x=(a!=foo);
        va_end(ap);
        return x;
}

void exit(); // expected-warning {{a function declaration without a prototype is deprecated in all versions of C}}

int main(argc,argv) // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
        int argc;char**argv;
{
        exit(test("",foo));
}

