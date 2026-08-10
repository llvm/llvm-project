// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// PR5518
struct A { 
  explicit operator int(); // expected-note{{conversion to integral type}}
};

void x() { 
  switch(A()) { // expected-error{{explicit conversion to}}
  }
}
