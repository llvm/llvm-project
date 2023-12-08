// RUN: %clang_cc1 -verify -Wno-covered-switch-default %s
// expected-no-diagnostics

enum E {
    one,
    two,
    three,
    four
};


int test(enum E e) 
{
    switch (e) 
    {
        case one:
            return 7;
        case two ... two + 1:
            return 42;
        case four:
            return 25;
        default:
            return 0;
    }
}
