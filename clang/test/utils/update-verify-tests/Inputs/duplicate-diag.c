void foo() {
    // expected-error@+1{{use of undeclared identifier 'a'}}
    a = 2; a = 2;
    b = 2; b = 2;
    // expected-error@+1 3{{use of undeclared identifier 'c'}}
    c = 2; c = 2;
    // expected-error 2{{asdf}}
}
