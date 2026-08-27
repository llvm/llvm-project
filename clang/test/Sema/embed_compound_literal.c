// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -fsyntax-only -verify=ok -x c++ -Wno-c23-extensions %s
// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify -DNEGATIVE %s
// ok-no-diagnostics

char *p1 = (char[]){
#embed __FILE__
};

int *p2 = (int[]){
#embed __FILE__
};

int *p3 = (int[30]){
#embed __FILE__ limit(30)
};


#ifdef NEGATIVE

// Pointer element type
char *bad_ptr = (int (*[5]) 0){ // expected-error {{expected ')'}}
                                  // expected-note@-1 {{to match this '('}}
// expected-error@-2 {{incompatible pointer types initializing}}
// expected-error@-3 {{initializer element is not a compile-time constant}}
#embed __FILE__
// expected-error@-1 {{incompatible integer to pointer conversion}}
// expected-warning@-2 {{excess elements in array initializer}}
};


// Struct element type
struct S { int x; };
struct S bad_struct = (struct S[5]){ // expected-error {{initializing 'struct S'}}

#embed __FILE__
// expected-warning@-1 {{excess elements in array initializer}}
};


// Enum element type
enum E { A };
enum E bad_enum = (enum E[5]){ // expected-error {{incompatible pointer to integer conversion}}
// expected-error@-1 {{initializer element is not a compile-time constant}}

#embed __FILE__
// expected-warning@-1 {{excess elements in array initializer}}
};

#endif
