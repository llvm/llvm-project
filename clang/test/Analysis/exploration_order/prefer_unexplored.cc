// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -analyzer-config exploration_strategy=unexplored_first -analyzer-output=text -verify %s | FileCheck %s
// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -analyzer-config exploration_strategy=unexplored_first_queue -analyzer-output=text -verify %s | FileCheck %s

extern int coin();

int foo() {
    int *x = 0; // expected-note {{'x' initialized to a null pointer value}}
    while (coin()) { // expected-note{{Loop condition is true}}
        if (coin())  // expected-note {{Taking true branch}}
            return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
                       // expected-note@-1{{Dereference of null pointer (loaded from variable 'x')}}
    }
    return 0;
}

void bar() {
    while(coin()) // expected-note{{Loop condition is true}}
        if (coin()) // expected-note {{Assuming the condition is true}}
            foo(); // expected-note{{Calling 'foo'}}
}

int foo2() {
    int *x = 0; // expected-note {{'x' initialized to a null pointer value}}
    while (coin()) { // expected-note{{Loop condition is true}}
        if (coin())  // expected-note {{Taking false branch}}
            return false;
        else
            return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
                       // expected-note@-1{{Dereference of null pointer (loaded from variable 'x')}}
    }
    return 0;
}

void bar2() {
    while(coin()) // expected-note{{Loop condition is true}}
        if (coin()) // expected-note {{Assuming the condition is false}}
          return false;
        else
            foo(); // expected-note{{Calling 'foo'}}
}
