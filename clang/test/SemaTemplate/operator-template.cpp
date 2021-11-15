// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s

// Make sure we accept this
template<class X>struct A{typedef X Y;};
template<class X>bool operator==(A<X>,typename A<X>::Y); // expected-note{{candidate template ignored: could not match 'A<type-parameter-0-0>' against 'B<int> *'}}

int a(A<int> x) { return operator==(x,1); }

int a0(A<int> x) { return x == 1; }

// FIXME: the location information for the note isn't very good
template<class X>struct B{typedef X Y;};
template<class X>bool operator==(B<X>*,typename B<X>::Y); // \
// expected-error{{overloaded 'operator==' must have at least one parameter of class or enumeration type}} \
// expected-note{{candidate template ignored: substitution failure [with X = int]}}
int a(B<int> x) { return operator==(&x,1); } // expected-error{{no matching function for call to 'operator=='}} \
// expected-note{{in instantiation of function template specialization}}

// Ensure we take parameter list reversal into account in partial oredring.
namespace CompareOrdering {
  template<typename T> struct A {};
  template<typename T> int operator<=>(A<T>, int) = delete;
  template<typename T> int operator<=>(int, A<T*>);
  // OK, selects the more-specialized reversed function.
  bool b = A<int*>() < 0;
}
