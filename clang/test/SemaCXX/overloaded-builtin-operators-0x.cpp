// RUN: %clang_cc1 -fsyntax-only -fshow-overloads=best -std=c++11 -verify %s 
// expected-no-diagnostics

template <class T>
struct X
{
   operator T() const {return T();}
};

void test_char16t(X<char16_t> x) {
   bool b = x == char16_t();
}
