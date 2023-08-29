// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral -Wformat-non-iso -fblocks -std=c++11 %s

__attribute__((format(scanf, 1, 2)))
int scanf(const char *, ...);

template<typename... Args>
__attribute__((format(scanf, 1, 2)))
int scan(const char *fmt, Args &&...args) { // expected-warning{{GCC requires a function with the 'format' attribute to be variadic}}
    return scanf(fmt, args...);
}

union bag {
    bool b;
    unsigned char uc;
    signed char sc;
    unsigned short us;
    signed short ss;
    unsigned int ui;
    signed int si;
    unsigned long ul;
    signed long sl;
    unsigned long long ull;
    signed long long sll;
    __fp16 f16;
    float ff;
    double fd;
    long double fl;
};

void test(void) {
    bag b;
    scan("%hhi %hhu %hhi %hhu", &b.sc, &b.uc, &b.b, &b.b);
    scan("%hi %hu", &b.ss, &b.us);
    scan("%i %u", &b.si, &b.ui);
    scan("%li %lu", &b.sl, &b.ul);
    scan("%lli %llu", &b.sll, &b.ull);
    scan("%f", &b.ff);
    scan("%lf", &b.fd);
    scan("%Lf", &b.fl);

    // expected-warning@+4{{format specifies type 'short *' but the argument has type 'signed char *'}}
    // expected-warning@+3{{format specifies type 'unsigned short *' but the argument has type 'unsigned char *'}}
    // expected-warning@+2{{format specifies type 'short *' but the argument has type 'bool *'}}
    // expected-warning@+1{{format specifies type 'unsigned short *' but the argument has type 'bool *'}}
    scan("%hi %hu %hi %hu", &b.sc, &b.uc, &b.b, &b.b);

    // expected-warning@+3{{format specifies type 'long *' but the argument has type 'short *'}}
    // expected-warning@+2{{format specifies type 'char *' but the argument has type 'short *'}}
    // expected-warning@+1{{format specifies type 'int *' but the argument has type 'short *'}}
    scan("%hhi %i %li", &b.ss, &b.ss, &b.ss);

    // expected-warning@+3{{format specifies type 'float *' but the argument has type '__fp16 *'}}
    // expected-warning@+2{{format specifies type 'float *' but the argument has type 'double *'}}
    // expected-warning@+1{{format specifies type 'float *' but the argument has type 'long double *'}}
    scan("%f %f %f", &b.f16, &b.fd, &b.fl);

    // expected-warning@+3{{format specifies type 'double *' but the argument has type '__fp16 *'}}
    // expected-warning@+2{{format specifies type 'double *' but the argument has type 'float *'}}
    // expected-warning@+1{{format specifies type 'double *' but the argument has type 'long double *'}}
    scan("%lf %lf %lf", &b.f16, &b.ff, &b.fl);

    // expected-warning@+3{{format specifies type 'long double *' but the argument has type '__fp16 *'}}
    // expected-warning@+2{{format specifies type 'long double *' but the argument has type 'float *'}}
    // expected-warning@+1{{format specifies type 'long double *' but the argument has type 'double *'}}
    scan("%Lf %Lf %Lf", &b.f16, &b.ff, &b.fd);
}
