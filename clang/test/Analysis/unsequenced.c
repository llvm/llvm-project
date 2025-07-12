// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core

static void binary_op_checks_inline(void) {
    int a1 = 0;
    int b = 0;
    int* ip;
    
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

    b = a1 + b + (a1++);
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is read from here}}
    // expected-note@-4{{variable is written to here}}
    
    b = (a1++) - a1;
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}
    
    if (a1 + (a1++) == b) {}
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

    *(ip + a1) = a1++;
    // expected-warning@-1{{unsequenced modification and access}}
    // expected-warning@-2{{unsequenced write to and read from variable}}
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}
    
    (void)a1; (void)b;
}

static void* id(void* param) {
    return param;
}

static void binary_op_checks_with_functions(void) {
    int a = 0;

    *(int*)id(&a) = a++; // Equivalent to a = (a++);
    // expected-warning@-1{{unsequenced writes to variable}}
    // expected-note@-2{{variable is written to here}}
    // expected-note@-3{{variable is written to here}}

    *(int*)id(&a) = a + 1; // Equivalent to a = a + 1;
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
    // expected-note@-3{{variable is written to here}}
    // expected-note@-4{{variable is read from here}}
}

static int* inc_and_return(int* a) {
    ++*a;
    return a;
}

static void function_checks(void) {
    
}
