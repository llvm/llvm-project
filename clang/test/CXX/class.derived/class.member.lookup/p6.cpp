// RUN: %clang_cc1 -fsyntax-only -verify %s

class V { 
public: 
  int f(); 
  int x; 
};

class W { 
public: 
  int g(); // expected-note{{member found by ambiguous name lookup}}
  int y; // expected-note{{member found by ambiguous name lookup}}
};

class B : public virtual V, public W
{
public:
  int f(); 
  int x;
  int g();  // expected-note{{member found by ambiguous name lookup}}
  int y; // expected-note{{member found by ambiguous name lookup}}
};

class C : public virtual V, public W { };

class D : public B, public C { void glorp(); };

void D::glorp() {
  x++;
  f();
  y++; // expected-error{{member 'y' found in multiple base classes of different types}}
  g(); // expected-error{{member 'g' found in multiple base classes of different types}}
}

// PR6462
struct BaseIO { BaseIO* rdbuf() { return 0; } };
struct Pcommon : virtual BaseIO { int rdbuf() { return 0; } };
struct P : virtual BaseIO, Pcommon {};

void f() { P p; p.rdbuf(); }
