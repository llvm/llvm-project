// RUN: %clang_cc1 -fsyntax-only -verify -Wno-c23-extensions %s

struct S {
  int a, b;
};

void f(int x) {
  int *a = new int[2]{
#embed __FILE__ limit(4)
  // expected-error@-1 {{excess elements in array initializer}}
  };

  int *b = new int[4]{
#embed __FILE__ limit(4)
  };

  int *c = new int[8]{
#embed __FILE__ limit(4)
  };

  int *d = new int[x]{
#embed __FILE__ limit(4)
  };

  int (*e)[2] = new int[2][2]{
#embed __FILE__ limit(5)
  // expected-error@-1 {{excess elements in array initializer}}
  };

  S *s = new S[1]{
#embed __FILE__ limit(3)
  // expected-error@-1 {{excess elements in array initializer}}
  };

  S *t = new S[x]{
#embed __FILE__ limit(3)
  };
}
