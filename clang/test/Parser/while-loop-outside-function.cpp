// RUN: %clang_cc1 -fsyntax-only -verify %s

while // expected-error {{while loop outside of a function}}
(true) {};

// without semicolon
while // expected-error {{while loop outside of a function}}
(true) {}

do { // expected-error {{expected unqualified-id}}
    int some_var = 1;
    some_var += 3;
} 
while // expected-error {{while loop outside of a function}}
(true); 

void someFunction() {
    while(true) {};
}

class SomeClass {
public:
    while(true) {} // expected-error {{expected member name or ';' after declaration specifiers}}
    void some_fn() {
        while(true) {}
    }
};
