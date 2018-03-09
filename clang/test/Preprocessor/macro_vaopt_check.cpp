// RUN: %clang_cc1 %s -Eonly -verify -Wno-all -pedantic -std=c++2a

//expected-error@+1{{missing '('}}
#define V1(...) __VA_OPT__  
#undef V1
// OK
#define V1(...) __VA_OPT__  ()
#undef V1 

//expected-warning@+1{{can only appear in the expansion of a variadic macro}}
#define V2() __VA_OPT__(x) 
#undef V2

//expected-error@+2{{missing ')' after}}
//expected-note@+1{{to match this '('}}
#define V3(...) __VA_OPT__(
#undef V3

#define V4(...) __VA_OPT__(__VA_ARGS__)
#undef V4

//expected-error@+1{{nested}}
#define V5(...) __VA_OPT__(__VA_OPT__())
#undef V5

//expected-error@+1{{not followed by}}
#define V1(...) __VA_OPT__  (#)
#undef V1

//expected-error@+1{{cannot appear at start}}
#define V1(...) __VA_OPT__  (##)
#undef V1

//expected-error@+1{{cannot appear at start}}
#define V1(...) __VA_OPT__  (## X) x
#undef V1

//expected-error@+1{{cannot appear at end}}
#define V1(...) y __VA_OPT__  (X ##)
#undef V1
                            

#define FOO(x,...) # __VA_OPT__(x) #x #__VA_OPT__(__VA_ARGS__) //OK

//expected-error@+1{{not followed by a macro parameter}}
#define V1(...) __VA_OPT__(#)
#undef V1

//expected-error@+1{{cannot appear at start}}
#define V1(...) a __VA_OPT__(##) b
#undef V1

//expected-error@+1{{cannot appear at start}}
#define V1(...) a __VA_OPT__(a ## b) b __VA_OPT__(##)
#undef V1

#define V1(x,...) # __VA_OPT__(b x) // OK
#undef V1

//expected-error@+2{{missing ')' after}}
//expected-note@+1{{to match this '('}}
#define V1(...) __VA_OPT__  ((())
#undef V1

