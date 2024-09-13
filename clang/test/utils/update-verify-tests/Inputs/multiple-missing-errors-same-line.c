void foo() {
    a = 2; b = 2; c = 2;
}

void bar() {
    x = 2; y = 2; z = 2;
    // expected-error@-1{{use of undeclared identifier 'x'}}
}
