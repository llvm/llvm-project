// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

class C {
public:
        enum E { e1=0 };
        const char * fun1(int , enum E) const;
        int fun1(unsigned, const char *) const;
};

void foo(const C& rc) {
        enum {BUFLEN = 128 };
        const char *p = rc.fun1(BUFLEN - 2, C::e1);
}
