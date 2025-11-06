// RUN: %clang_cc1 -Wfree-nonheap-object -fsyntax-only -verify %s

void c1(int *a);
typedef __typeof__(sizeof(0)) size_t;
extern int g1 __attribute((cleanup(c1))); // expected-warning {{'cleanup' attribute only applies to local variables}}
int g2 __attribute((cleanup(c1))); // expected-warning {{'cleanup' attribute only applies to local variables}}
static int g3 __attribute((cleanup(c1))); // expected-warning {{'cleanup' attribute only applies to local variables}}

void t1(void)
{
    int v1 __attribute((cleanup)); // expected-error {{'cleanup' attribute takes one argument}}
    int v2 __attribute((cleanup(1, 2))); // expected-error {{'cleanup' attribute takes one argument}}

    static int v3 __attribute((cleanup(c1))); // expected-warning {{'cleanup' attribute only applies to local variables}}

    int v4 __attribute((cleanup(h))); // expected-error {{use of undeclared identifier 'h'}}

    int v5 __attribute((cleanup(c1)));
    int v6 __attribute((cleanup(v3))); // expected-error {{'cleanup' argument 'v3' is not a function}}
}

struct s {
    int a, b;
};

void c2(void);
void c3(struct s a);

void t2(void)
{
    int v1 __attribute__((cleanup(c2))); // expected-error {{'cleanup' function 'c2' must take 1 parameter}}
    int v2 __attribute__((cleanup(c3))); // expected-error {{'cleanup' function 'c3' parameter has type 'struct s' which is incompatible with type 'int *'}}
}

// This is a manufactured testcase, but gcc accepts it...
void c4(_Bool a);
void t4(void) {
  __attribute((cleanup(c4))) void* g;
}

void c5(void*) __attribute__((deprecated));  // expected-note{{'c5' has been explicitly marked deprecated here}}
void t5(void) {
  int i __attribute__((cleanup(c5)));  // expected-warning {{'c5' is deprecated}}
}

void t6(void) {
  int i __attribute__((cleanup((void *)0)));  // expected-error {{'cleanup' argument is not a function}}
}

void t7(__attribute__((cleanup(c4))) int a) {} // expected-warning {{'cleanup' attribute only applies to local variables}}

extern void free(void *);
extern void *malloc(size_t size);
void t8(void) {
  void *p
  __attribute__((
	  cleanup(
		  free // expected-warning{{attempt to call free on non-heap object 'p'}}
	  )
	))
  = malloc(10);
}
typedef __attribute__((aligned(2))) int Aligned2Int;
void t9(void){
  Aligned2Int __attribute__((cleanup(c1))) xwarn; // expected-warning{{passing 2-byte aligned argument to 4-byte aligned parameter 1 of 'c1' may result in an unaligned pointer access}}
}

__attribute__((enforce_tcb("TCB1"))) void func1(int *x) {
    *x = 5;
}
__attribute__((enforce_tcb("TCB2"))) void t10() {
  int __attribute__((cleanup(func1))) x = 5; // expected-warning{{calling 'func1' is a violation of trusted computing base 'TCB2'}}
}

