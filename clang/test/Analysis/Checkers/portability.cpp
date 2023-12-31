// RUN: %clang_analyze_cc1 -analyzer-checker=optin.portability -verify %s

extern int printf(const char *, ...);

void print_null_ptr() {
    int x = 0;
    printf("ppppp%dppppp", x); // no warning

    int* p = &x;
    printf("dddd%pddddd", p); // no warning
    
    p = nullptr;
    printf("dddddd%pdddddd", p); // expected-warning{{Output null pointer with printf %p}}
}

void test2() {
    int x = 0;
    void* p = &x;
    printf("%d %p", x, p); // no warning

    p = nullptr;
    printf("%d %p", x, p); // expected-warning{{Output null pointer with printf %p}}
}
