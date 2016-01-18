// RUN: %clang_cc1 -fsyntax-only -verify %s 
class Z { };

class Y { 
public:
  Y(const Z&);
};

class X {
public:
  X(int);
  X(const Y&);
};

void f(X); // expected-note{{candidate function}}

void g(short s, Y y, Z z) {
  f(s);
  f(1.0f);
  f(y);
  f(z); // expected-error{{no matching function}}
}


class FromShort {
public:
  FromShort(short s);
};

class FromShortExplicitly { // expected-note{{candidate constructor (the implicit copy constructor)}}
public:
  explicit FromShortExplicitly(short s);
};

void explicit_constructor(short s) {
  FromShort fs1(s);
  FromShort fs2 = s;
  FromShortExplicitly fse1(s);
  FromShortExplicitly fse2 = s; // expected-error{{no viable conversion}}
}

// PR5519
struct X1 { X1(const char&); };
void x1(X1);
void y1() {
  x1(1);
}
