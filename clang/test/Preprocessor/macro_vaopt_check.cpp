// RUN: %clang_cc1 %s -Eonly -verify -Wno-all -Wno-c++2b-extensions -pedantic -std=c++20
// RUN: %clang_cc1 %s -Eonly -verify -Wno-all -Wno-c++2b-extensions -pedantic -std=c++11
// RUN: %clang_cc1 -x c %s -Eonly -verify -Wno-all -Wno-c2x-extensions -pedantic -std=c99

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

// __VA_OPT__ can't appear anywhere else.
#if __VA_OPT__ // expected-warning {{__VA_OPT__ can only appear in the expansion of a variadic macro}}
#endif

// expected-warning@+2 {{__VA_OPT__ can only appear in the expansion of a variadic macro}}
#ifdef __VA_OPT__ // expected-warning {{__VA_OPT__ can only appear in the expansion of a variadic macro}}
#elifdef __VA_OPT__
#endif

#define BAD __VA_OPT__ // expected-warning {{__VA_OPT__ can only appear in the expansion of a variadic macro}}
