// expected-error@+1{{expected (}}
#pragma clang restrict_expansion

// expected-error@+1{{expected identifier}}
#pragma clang restrict_expansion(4

// expected-error@+1{{no macro named 'foo'}}
#pragma clang restrict_expansion(foo)


#define UNSAFE_MACRO 1
// expected-note@+8{{macro marked 'restrict_expansion' here}}
// expected-note@+7{{macro marked 'restrict_expansion' here}}
// expected-note@+6{{macro marked 'restrict_expansion' here}}
// expected-note@+5{{macro marked 'restrict_expansion' here}}
// expected-note@+4{{macro marked 'restrict_expansion' here}}
// expected-note@+3{{macro marked 'restrict_expansion' here}}
// expected-note@+2{{macro marked 'restrict_expansion' here}}
// expected-note@+1{{macro marked 'restrict_expansion' here}} 
#pragma clang restrict_expansion(UNSAFE_MACRO, "Don't use this!")

#define UNSAFE_MACRO_2 2
// expected-note@+1{{macro marked 'restrict_expansion' here}}
#pragma clang restrict_expansion(UNSAFE_MACRO_2)

// expected-error@+1{{expected )}}
#pragma clang deprecated(UNSAFE_MACRO
