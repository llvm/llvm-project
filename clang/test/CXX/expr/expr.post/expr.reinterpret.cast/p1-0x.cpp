// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// If T is an lvalue reference type or an rvalue reference to function
// type, the result is an lvalue; if T is an rvalue reference to
// object type, the result is an xvalue;

unsigned int f(int);

template<typename T> T&& xvalue();
void test_classification(char *ptr) {
  int (&fr0)(int) = reinterpret_cast<int (&&)(int)>(f);
  int &&ir0 = reinterpret_cast<int &&>(*ptr);
  int &&ir1 = reinterpret_cast<int &&>(0); // expected-error {{rvalue to reference type}}
  int &&ir2 = reinterpret_cast<int &&>('a'); // expected-error {{rvalue to reference type}}
  int &&ir3 = reinterpret_cast<int &&>(xvalue<char>());
  // Per DR1268, reinterpret_cast can convert between lvalues and xvalues.
  int &ir4 = reinterpret_cast<int &>(xvalue<char>());
  int &&ir5 = reinterpret_cast<int &&>(*ptr);
}
