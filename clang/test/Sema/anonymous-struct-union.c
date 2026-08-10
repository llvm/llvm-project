// RUN: %clang_cc1 -fsyntax-only -verify %s
struct X {
  union {
    float f3;
    double d2;
  } named;

  union {
    int i;
    float f;
    
    union {
      float f2;
      double d;
    };
  };

  struct {
    int a;
    float b;
  };
};

void test_unqual_references(struct X x, const struct X xc) {
  // expected-note@-1 3{{variable 'xc' declared const here}}
  x.i = 0;
  x.f = 0.0;
  x.f2 = x.f;
  x.d = x.f;
  x.f3 = 0; // expected-error{{no member named 'f3'}}
  x.a = 0;

  xc.d = 0.0; // expected-error{{cannot assign to variable 'xc' with const-qualified type 'const struct X'}}
  xc.f = 0; // expected-error{{cannot assign to variable 'xc' with const-qualified type 'const struct X'}}
  xc.a = 0; // expected-error{{cannot assign to variable 'xc' with const-qualified type 'const struct X'}}
}


struct Redecl {
  int x; // expected-note{{previous declaration is here}}
  struct y { }; // expected-warning{{declaration does not declare anything}}

  union {
    int x; // expected-error{{member of anonymous union redeclares 'x'}}
    float y;
    double z; // expected-note{{previous declaration is here}}
    double zz; // expected-note{{previous declaration is here}}
  };

  int z; // expected-error{{duplicate member 'z'}}
  void zz(void); // expected-error{{duplicate member 'zz'}} 
};

union { // expected-warning{{declaration does not declare anything}}
  int int_val;
  float float_val;
};

static union { // expected-warning{{declaration does not declare anything}}
  int int_val2;
  float float_val2;
};

void f(void) {
  int_val2 = 0; // expected-error{{use of undeclared identifier}}
  float_val2 = 0.0; // expected-error{{use of undeclared identifier}}
}

void g(void) {
  union { // expected-warning{{declaration does not declare anything}}
    int i;
    float f2;
  };
  i = 0; // expected-error{{use of undeclared identifier}}
  f2 = 0.0; // expected-error{{use of undeclared identifier}}
}

struct s0 { union { int f0; }; };

typedef struct { }; // expected-warning{{typedef requires a name}}

// PR3675
struct s1 {
  int f0; // expected-note{{previous declaration is here}}
  union {
    int f0; // expected-error{{member of anonymous union redeclares 'f0'}}
  };
};

// PR3680
struct {}; // expected-warning{{declaration does not declare anything}}

struct s2 {
  union {
    int a;
  } // expected-warning{{expected ';' at end of declaration list}}
}; // expected-error{{expected member name or ';' after declaration specifiers}}

// Make sure we don't a.k.a. anonymous structs.
typedef struct {
  int x;
} a_struct;
int tmp = (a_struct) { .x = 0 }; // expected-error {{initializing 'int' with an expression of incompatible type 'a_struct'}}

// This example comes out of the C11 standard; make sure we don't accidentally reject it.
struct s {
  struct { int i; };
  int a[];
};

// PR20930
struct s3 {
  struct { int A __attribute__((deprecated)); }; // expected-note {{'A' has been explicitly marked deprecated here}}
};

void deprecated_anonymous_struct_member(void) {
  struct s3 s;
  s.A = 1; // expected-warning {{'A' is deprecated}}
}
