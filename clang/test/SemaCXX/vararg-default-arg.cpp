// RUN: %clang_cc1 %s -verify -fsyntax-only
// expected-no-diagnostics
// PR5462

void f1(void);
void f2(const char * = __null, ...);

void f1(void)
{
        f2();
}
