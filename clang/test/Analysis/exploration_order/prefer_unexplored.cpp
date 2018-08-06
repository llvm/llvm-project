// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -analyzer-config exploration_strategy=unexplored_first -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -analyzer-config exploration_strategy=unexplored_first_queue -analyzer-output=text -verify %s

extern int coin();

int foo() {
    int *x = 0; // expected-note {{'x' initialized to a null pointer value}}
    while (coin()) { // expected-note{{Loop condition is true}}
        if (coin())  // expected-note {{Taking true branch}}
                     // expected-note@-1 {{Assuming the condition is true}}
            return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
                       // expected-note@-1{{Dereference of null pointer (loaded from variable 'x')}}
    }
    return 0;
}

void bar() {
    while(coin())
        if (coin())
            foo();
}

int foo2() {
    int *x = 0; // expected-note {{'x' initialized to a null pointer value}}
    while (coin()) { // expected-note{{Loop condition is true}}
        if (coin())  // expected-note {{Taking false branch}}
                     // expected-note@-1 {{Assuming the condition is false}}
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
                    // expected-note@-1 {{Taking false branch}}
          return;
        else
            foo(); // expected-note{{Calling 'foo'}}
}
