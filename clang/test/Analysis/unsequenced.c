// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core

static void binary_op_checks_inline(void) {
    int a1 = 0;
    int b = 0;
    int* ip;
    int arr[10];
    
    b = a1 + (a1++);
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}

    a1 = a1++;
    // expected-warning@-1{{multiple unsequenced modifications}}
    // expected-warning@-2{{unsequenced writes to variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is written to here}}
    
    a1 = ++a1;
    // expected-warning@-1{{multiple unsequenced modifications}}
    // expected-warning@-2{{unsequenced writes to variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is written to here}}
    
    a1 = a1;

    b = a1 + b + (++a1);
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    
    b = (a1++) - a1;
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}
    
    if (a1 + (a1 = 0) == b) {}
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}

    while (a1 + a1 == b) {}
    a1 = a1 + 1;
    a1 += a1;
    b = (a1++, a1);

    ip = &b;

    if (*ip + (a1++)) {}
    
    ip = &a1;

    if (*ip == (a1++)) {}
    // expected-warning@-1{{unsequenced write to and read from variable}}
    // expected-note@-2{{variable is read from here}}
    // expected-note@-3{{variable is written to here}}

    *(ip + a1) = a1 -= 1;
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}

    // Array subscript expressions are essentially just addition

    (arr + a1)[a1++] = 0;
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}
    
    (arr + --a1)[a1] = 1;
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}

    if (((*ip = 1 || b == 2) ? 0 : 3) + a1) {}
    // expected-warning@-1{{unsequenced write to and read from variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is read from here}}

    (arr + sizeof(a1))[1] = a1++;

    b = a1 + (long)&a1;

    b = a1 << (++a1);
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}

    *ip = a1 = a1 + 1;
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}

    a1 = 2 + (a1 = 0);
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}
    // expected-warning@-4{{multiple unsequenced modifications}}
    
    a1 = a1 = 1;
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}
    // expected-warning@-4{{multiple unsequenced modifications}}
    
    (*&a1) = ++a1;
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}

    (*&a1) = a1++;
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}

    (void)a1; (void)b;
}

static void* id(void* param) {
    return param;
}

static int id_i(int param) {
    return param;
}

static void binary_op_checks_with_functions(void) {
    int a = 0;

    *(int*)id(&a) = a++; // Equivalent to a = (a++);
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}

    *(int*)id(&a) = a + 1; // Equivalent to a = a + 1;
    
    a = id_i(a++);
}

static void does_nothing2(int a, int b) {
}

static void function_checks_inline(void) {
    int a = 0;

    does_nothing2(a, a++);
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    
    does_nothing2(a++, a++);
    // expected-warning@-1{{multiple unsequenced modifications}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-warning@-3{{unsequenced writes to variable}}
    // expected-note@-4{{variable is written to here}}
    // expected-note@-5{{variable is read from here}}
    // expected-note@-6{{variable is written to here}}
    // expected-note@-7{{variable is written to here}}

    does_nothing2(a = 0, a = 1);
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}
    // expected-warning@-4{{multiple unsequenced modifications}}
}

static int* inc_and_return(int* a) {
    ++*a;
    // expected-note@-1{{variable is written to here}}
    return a;
}

int globalInt;

typedef int (*some_fun_t)(int);

static int some_fun_impl(int x) {
    return 0;
}

static int inc_global_int(int a) {
    globalInt++;
    return a + 1;
}

static void function_checks(void) {

    float a2 = 0.1f;
    float* fp = &a2;
    int a3 = 22;

    some_fun_t some_funs[3] = {0};

    if (inc_and_return(&globalInt) + globalInt) {}
    // expected-warning@-1{{unsequenced write to and read from variable}}
    // expected-note@-2{{variable is read from here}}
    
    if (inc_and_return(&globalInt) || globalInt) {}

    a2 = ++a2;
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}
    // expected-warning@-4{{multiple unsequenced modifications}}
    
    *fp = ++*fp;
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}
    
    a2 = ++*fp;
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}

    a3 = *inc_and_return(&a3);

    some_funs[1] = some_fun_impl;

    a3 = 1;
    some_funs[a3](a3++);
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    
    if (inc_global_int(globalInt) == 0) {}

}

static void loop_checks(void) {
    char* str[3] = {0, 0, 0};

    char c = 'c';

    unsigned i;

    str[2] = &c;

    for (i = 0; i < sizeof(str) / sizeof(*str); i++) {
        if (str[i]) {
            *(str[i]) = c++;
            // expected-warning@-1{{unsequenced writes to variable}}
            // expected-note@-2{{variable is written to here}}
            // expected-note@-3{{variable is written to here}}
        }
    }
}

int globalI1;
int globalI2;

static int incGlobalI1(void) {
    return ++globalI1;
    // expected-note@-1{{variable is written to here}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}
}

static int incGlobalI2(void) {
    return ++globalI2;
}

static int branch1(void) {
    return incGlobalI1();
}

static int branch2(void) {
    return incGlobalI2(), incGlobalI1();
}

static int branching_function_check(void) {
    return branch1() + branch2();
    // expected-warning@-1{{unsequenced write to and read from variable}}
    // expected-warning@-2{{unsequenced writes to variable}}
}

static int function_with_auto_lifetimes(void) {
    int a = 0;
    int b[2] = {0};
    float c = 2;

    a++;
    b[1]--;
    c = c + a;
    b[--a] = c;

    return b[0];
}

static int unsequenced_but_no_ub(void) {
    return function_with_auto_lifetimes() + function_with_auto_lifetimes();
}

static unsigned function_with_static_lifetimes(void) {
    static unsigned a = 0;
    return a++;
    // expected-note@-1{{variable is read from here}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is written to here}}
}

static unsigned unsequenced_with_ub(void) {
    return function_with_static_lifetimes() - function_with_static_lifetimes();
    // expected-warning@-1{{unsequenced write to and read from variable}}
    // expected-warning@-2{{unsequenced writes to variable}}
}

struct int_wrapper {
    int data;
};

int post_inc_wrapper(struct int_wrapper* wrap) {
    wrap->data += 1;
    // expected-note@-1{{variable is written to here}}
    return wrap->data;
}

static void dst_tests(void) {
    struct int_wrapper wrap = {0};

    if (wrap.data + post_inc_wrapper(&wrap)) {}
    // expected-warning@-1{{unsequenced write to and read from variable}}
    // expected-note@-2{{variable is read from here}}
}
