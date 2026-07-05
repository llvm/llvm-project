// RUN: %clang_cc1 -verify -std=c++23 -Wall -Wshadow %s

void f() {

    for (using foo = int;true;) {} //expected-warning {{unused type alias 'foo'}}

    switch(using foo = int; 0) { //expected-warning {{unused type alias 'foo'}}
        case 0: break;
    }

    if(using foo = int; false) {} // expected-warning {{unused type alias 'foo'}}

    int x; // expected-warning {{unused variable 'x'}}
    if(using x = int; true) {}  // expected-warning {{unused type alias 'x'}}

    using y = int; // expected-warning {{unused type alias 'y'}} \
                   // expected-note 2{{previous declaration is here}}

    if(using y = double; true) {}  // expected-warning {{unused type alias 'y'}} \
                                   // expected-warning {{declaration shadows a type alias in function 'f'}}

    for(using y = double; true;) { // expected-warning {{declaration shadows a type alias in function 'f'}}
        y foo = 0;
        (void)foo;
        constexpr y var = 0;
        static_assert(var == 0);
    }
}
