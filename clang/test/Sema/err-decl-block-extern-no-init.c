// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
static int x;

void foo(void)
{
    extern int x = 1; // expected-error {{declaration of block scope identifier with linkage cannot have an initializer}}
}

int y;

void bar(void)
{
    extern int y = 1; // expected-error {{declaration of block scope identifier with linkage cannot have an initializer}}

}
