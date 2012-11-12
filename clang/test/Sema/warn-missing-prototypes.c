// RUN: %clang -Wmissing-prototypes -fsyntax-only -Xclang -verify %s

int f();

int f(int x) { return x; } // expected-warning{{no previous prototype for function 'f'}}

static int g(int x) { return x; }

int h(int x) { return x; } // expected-warning{{no previous prototype for function 'h'}}

static int g2();

int g2(int x) { return x; }

void test(void);

int h3();
int h4(int);
int h4();

void test(void) {
  int h2(int x);
  int h3(int x);
  int h4();
}

int h2(int x) { return x; } // expected-warning{{no previous prototype for function 'h2'}}
int h3(int x) { return x; } // expected-warning{{no previous prototype for function 'h3'}}
int h4(int x) { return x; }

int f2(int);
int f2();

int f2(int x) { return x; }

// rdar://6759522
int main(void) { return 0; }
