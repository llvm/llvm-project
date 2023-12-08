// RUN: %clang_cc1 -fsyntax-only -verify %s

template<int i> class A {  };
template<short s> void f(A<s>); // expected-note{{candidate template ignored: substitution failure}}

void k1() { 
  A<1> a;
  f(a); // expected-error{{no matching function for call}}
  f<1>(a);
}
template<const short cs> class B { }; 
template<short s> void g(B<s>); 
void k2() {
  B<1> b; 
  g(b); // OK: cv-qualifiers are ignored on template parameter types
}

template<short s> void h(int (&)[s]); // expected-note{{candidate function template not viable: requires 1 argument, but 2 were provided}}
void k3() {
  int array[5];
  h(array);
  h<5>(array);
}

template<short s> void h(int (&)[s], A<s>);  // expected-note{{candidate template ignored: substitution failure}}
void k4() {
  A<5> a;
  int array[5];
  h(array, a); // expected-error{{no matching function for call}}
  h<5>(array, a);
}
