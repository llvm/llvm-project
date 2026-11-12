// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -DSYSTEM -verify %s
// RUN: %clang_cc1 -fsyntax-only -DNOERROR -Wno-error=init-priority-reserved -verify %s

#if defined(SYSTEM)
#5 "init-priority-attr.cpp" 3 // system header
#endif

class Two {
private:
    int i, j, k;
public:
    static int count;
    Two( int ii, int jj ) { i = ii; j = jj; k = count++; };
    Two( void )           { i =  0; j =  0; k = count++; };
    int eye( void ) { return i; };
    int jay( void ) { return j; };
    int kay( void ) { return k; };
};

extern Two foo;
extern Two goo;
extern Two coo[];
extern Two koo[];

// unknown-system-no-diagnostics

Two foo __attribute__((init_priority(101))) ( 5, 6 );

Two loo __attribute__((init_priority(65535))) ( 5, 6 );

Two goo __attribute__((init_priority(2,3))) ( 5, 6 ); // expected-error {{'init_priority' attribute takes one argument}}

Two coo[2]  __attribute__((init_priority(100)));
#if !defined(SYSTEM)
#if !defined(NOERROR)
  // expected-error@-3 {{requested 'init_priority' 100 is reserved for internal use}}
#else  // defined(NOERROR)
  // expected-warning@-5 {{requested 'init_priority' 100 is reserved for internal use}}
#endif // !defined(NOERROR)
#endif // !defined(SYSTEM)
Two zoo[2]  __attribute__((init_priority(-1))); // expected-error {{'init_priority' attribute requires integer constant between 0 and 65535 inclusive}}

Two boo[2]  __attribute__((init_priority(65536))); // expected-error {{'init_priority' attribute requires integer constant between 0 and 65535 inclusive}}

Two koo[4]  __attribute__((init_priority(1.13))); // expected-error {{'init_priority' attribute requires an integer constant}}

Two func()  __attribute__((init_priority(1001))); // expected-error {{'init_priority' attribute only applies to variables}}

int i  __attribute__((init_priority(1001))); // expected-error {{can only use 'init_priority' attribute on file-scope definitions of objects of class type}}

int main() {
  Two foo __attribute__((init_priority(1001))); // expected-error {{can only use 'init_priority' attribute on file-scope definitions of objects of class type}}
}

struct S1 {} s1;
[[gnu::init_priority(1001)]] auto auto_var = s1;
// unknown-warning@-1 {{unknown attribute 'gnu::init_priority' ignored}}
[[gnu::init_priority(1001)]] S1 struct_var = s1;
// unknown-warning@-1 {{unknown attribute 'gnu::init_priority' ignored}}
