// RUN: %clang_cc1 -fsyntax-only -verify %s

while // expected-error {{while loop outside of a function}}
(1) {};

// without semicolon
while // expected-error {{while loop outside of a function}}
(1) {}

int overload_return(); // expected-note {{previous declaration is here}}

void overload_return() // expected-error {{conflicting types for 'overload_return'}}
{ 
    while(1) {};
    while(1);
}

while // expected-error {{while loop outside of a function}}
(1); 

void correct();

void correct() {
    while(1) {};
    while(1);
}
