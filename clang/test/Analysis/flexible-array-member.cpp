// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++11 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

struct S
{
    static int c;
    static int d;
    int x;
    S() { x = c++; }
    ~S() { d++; }
};

int S::c = 0;
int S::d = 0;

struct Flex
{
    int length;
    S contents[0];
};

void flexibleArrayMember()
{
    S::c = 0;
    S::d = 0;

    const int size = 4;

    Flex *arr =
        (Flex *)::operator new(__builtin_offsetof(Flex, contents) + sizeof(S) * size);

    clang_analyzer_eval(S::c == 0); // expected-warning{{TRUE}}

    new (&arr->contents[0]) S;
    new (&arr->contents[1]) S;
    new (&arr->contents[2]) S;
    new (&arr->contents[3]) S;

    clang_analyzer_eval(S::c == size); // expected-warning{{TRUE}}

    clang_analyzer_eval(arr->contents[0].x == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(arr->contents[1].x == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(arr->contents[2].x == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(arr->contents[3].x == 3); // expected-warning{{TRUE}}

    arr->contents[0].~S();
    arr->contents[1].~S();
    arr->contents[2].~S();
    arr->contents[3].~S();

    ::operator delete(arr);

    clang_analyzer_eval(S::d == size); // expected-warning{{TRUE}}
}
