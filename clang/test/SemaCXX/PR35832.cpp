// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

class B {
public:
 int i;
 struct {  struct { union { int j; }; };  };
 union { int k; };
};

class X : public B { };
class Y : public B { };

class Z : public X, public Y {
public:
 int a() { return X::i; }
 int b() { return X::j; }
 int c() { return X::k; }
 int d() { return this->X::j; }
};
