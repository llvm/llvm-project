// RUN: %clang_analyze_cc1 -verify %s -std=c++11 -analyzer-checker=core
// RUN: %clang_analyze_cc1 -verify %s -std=c++17 -analyzer-checker=core
// RUN: %clang_analyze_cc1 -verify %s -analyzer-checker=core

#if __cplusplus >= 201703L
#   define CPP17
#endif

void does_nothing2(int, int) {}

void shift_tests() {
    int a = 1;
    int b;
    (void)b;

    b = a++ << a;
#ifndef CPP17
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{unsequenced modification and access}}
#endif

    if (a << (++a) != 0) {}
#ifndef CPP17
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{unsequenced modification and access}}
#endif

    int i = 0;
    if(a << (i || (i = a++)) == 0) {}
#ifndef CPP17
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{unsequenced modification and access}}
#endif

    a = a = 1;
}

void std_tests() {
    int a = 1;

    a = a++;
#ifndef CPP17
    // expected-warning@-2{{unsequenced writes to variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{multiple unsequenced modifications}}
#endif

    a = ++a; // No undefined behavior since C++11

    int s[3] = {0};
    int* sp = s;
    int j = 0;
    while(j < 3) (sp + j)[j++] = 1;
#ifndef CPP17
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{unsequenced modification and access}}
#endif
    
    int r[3];
    int i = 0;
    while (i < 3) r[i] = i++;
#ifndef CPP17
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{unsequenced modification and access}}
#endif

    i = 0;
    r[i] = ++i;
#ifndef CPP17
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{unsequenced modification and access}}
#endif
}

class IntWrapper {
    int data;
public:
    IntWrapper(int d) : data(d) {}

    // Prefix overload
    IntWrapper operator++() {
        data += 1;
        return IntWrapper(data);
    }

    // Postfix overload
    IntWrapper operator++(int) {
        int old_data = data;
        data += 1;
        return IntWrapper(old_data);
    }

    IntWrapper operator+(const IntWrapper& Other) {
        return IntWrapper(Other.data + data);
    }
};

void some_wrapper_func(IntWrapper, IntWrapper);

void object_tests() {
    IntWrapper A(0);

    IntWrapper I = A + A++;

    some_wrapper_func(A, A++);
}

typedef int (*some_fun_t)(int);

static int some_fun_impl(int x) {
    return 0;
}

void function_tests() {
    some_fun_t some_funs[2] = {0};
    some_funs[0] = some_fun_impl;
    int a = 0;
    some_funs[a](a++);
#ifndef CPP17
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    // expected-warning@-5{{unsequenced modification and access}}
#endif
}

#ifdef CPP17
    // expected-no-diagnostics
#endif
