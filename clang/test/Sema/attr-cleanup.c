// RUN: %clang_cc1 %s -verify -fsyntax-only

void c1(int *a);

extern int g1 __attribute((cleanup(c1))); // expected-warning {{cleanup attribute ignored}}
int g2 __attribute((cleanup(c1))); // expected-warning {{cleanup attribute ignored}}
static int g3 __attribute((cleanup(c1))); // expected-warning {{cleanup attribute ignored}}

void t1()
{
    int v1 __attribute((cleanup)); // expected-error {{attribute takes one argument}}
    int v2 __attribute((cleanup(1, 2))); // expected-error {{attribute takes one argument}}
    
    static int v3 __attribute((cleanup(c1))); // expected-warning {{cleanup attribute ignored}}
    
    int v4 __attribute((cleanup(h))); // expected-error {{'cleanup' argument 'h' not found}}

    int v5 __attribute((cleanup(c1)));
    int v6 __attribute((cleanup(v3))); // expected-error {{'cleanup' argument 'v3' is not a function}}
}

struct s {
    int a, b;
};

void c2();
void c3(struct s a);

void t2()
{
    int v1 __attribute__((cleanup(c2))); // expected-error {{'cleanup' function 'c2' must take 1 parameter}}
    int v2 __attribute__((cleanup(c3))); // expected-error {{'cleanup' function 'c3' parameter has type 'struct s' which is incompatible with type 'int *'}}
}

// This is a manufactured testcase, but gcc accepts it...
void c4(_Bool a);
void t4() {
  __attribute((cleanup(c4))) void* g;
}

