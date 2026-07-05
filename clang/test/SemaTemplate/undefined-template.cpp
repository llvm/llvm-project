// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 -Wundefined-func-template %s

#if !defined(INCLUDE)
template <class T> struct C1 {
  static char s_var_1;       // expected-note{{forward declaration of template entity is here}}
  static char s_var_2;       // expected-note{{forward declaration of template entity is here}}
  static void s_func_1();    // expected-note{{forward declaration of template entity is here}}
  static void s_func_2();    // expected-note{{forward declaration of template entity is here}}
  void meth_1();             // expected-note2{{forward declaration of template entity is here}}
  void meth_2();
  template <class T1> static char s_tvar_2;      // expected-note{{forward declaration of template entity is here}}
  template <class T1> static void s_tfunc_2();   // expected-note{{forward declaration of template entity is here}}
  template<typename T1> struct C2 {
    static char s_var_2;     // expected-note{{forward declaration of template entity is here}}
    static void s_func_2();  // expected-note{{forward declaration of template entity is here}}
    void meth_2();           // expected-note{{forward declaration of template entity is here}}
    template <class T2> static char s_tvar_2;    // expected-note{{forward declaration of template entity is here}}
    template <class T2> void tmeth_2();          // expected-note{{forward declaration of template entity is here}}
  };
};

extern template char C1<int>::s_var_2;
extern template void C1<int>::s_func_2();
extern template void C1<int>::meth_2();
extern template char C1<int>::s_tvar_2<char>;
extern template void C1<int>::s_tfunc_2<char>();
extern template void C1<int>::C2<long>::s_var_2;
extern template void C1<int>::C2<long>::s_func_2();
extern template void C1<int>::C2<long>::meth_2();
extern template char C1<int>::C2<long>::s_tvar_2<char>;
extern template void C1<int>::C2<long>::tmeth_2<char>();

char func_01() {
  return C1<int>::s_var_2;
}

char func_02() {
  return C1<int>::s_var_1; // expected-warning{{instantiation of variable 'C1<int>::s_var_1' required here, but no definition is available}}
                           // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::s_var_1' is explicitly instantiated in another translation unit}}
}

char func_03() {
  return C1<char>::s_var_2; // expected-warning{{instantiation of variable 'C1<char>::s_var_2' required here, but no definition is available}}
                            // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<char>::s_var_2' is explicitly instantiated in another translation unit}}
}

void func_04() {
  C1<int>::s_func_1(); // expected-warning{{instantiation of function 'C1<int>::s_func_1' required here, but no definition is available}}
                       // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::s_func_1' is explicitly instantiated in another translation unit}}
}

void func_05() {
  C1<int>::s_func_2();
}

void func_06() {
  C1<char>::s_func_2(); // expected-warning{{instantiation of function 'C1<char>::s_func_2' required here, but no definition is available}}
                        // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<char>::s_func_2' is explicitly instantiated in another translation unit}}
}

void func_07(C1<int> *x) {
  x->meth_1();  // expected-warning{{instantiation of function 'C1<int>::meth_1' required here, but no definition is available}}
                // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::meth_1' is explicitly instantiated in another translation unit}}
}

void func_08(C1<int> *x) {
  x->meth_2();
}

void func_09(C1<char> *x) {
  x->meth_1();  // expected-warning{{instantiation of function 'C1<char>::meth_1' required here, but no definition is available}}
                // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<char>::meth_1' is explicitly instantiated in another translation unit}}
}

char func_10() {
  return C1<int>::s_tvar_2<char>;
}

char func_11() {
  return C1<int>::s_tvar_2<long>; // expected-warning{{instantiation of variable 'C1<int>::s_tvar_2<long>' required here, but no definition is available}}
                                  // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::s_tvar_2<long>' is explicitly instantiated in another translation unit}}
}

void func_12() {
  C1<int>::s_tfunc_2<char>();
}

void func_13() {
  C1<int>::s_tfunc_2<long>(); // expected-warning{{instantiation of function 'C1<int>::s_tfunc_2<long>' required here, but no definition is available}}
                              // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::s_tfunc_2<long>' is explicitly instantiated in another translation unit}}
}

char func_14() {
  return C1<int>::C2<long>::s_var_2;
}

char func_15() {
  return C1<int>::C2<char>::s_var_2;  //expected-warning {{instantiation of variable 'C1<int>::C2<char>::s_var_2' required here, but no definition is available}}
                                      // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::C2<char>::s_var_2' is explicitly instantiated in another translation unit}}
}

void func_16() {
  C1<int>::C2<long>::s_func_2();
}

void func_17() {
  C1<int>::C2<char>::s_func_2(); // expected-warning{{instantiation of function 'C1<int>::C2<char>::s_func_2' required here, but no definition is available}}
                        // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::C2<char>::s_func_2' is explicitly instantiated in another translation unit}}
}

void func_18(C1<int>::C2<long> *x) {
  x->meth_2();
}

void func_19(C1<int>::C2<char> *x) {
  x->meth_2();   // expected-warning{{instantiation of function 'C1<int>::C2<char>::meth_2' required here, but no definition is available}}
                        // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::C2<char>::meth_2' is explicitly instantiated in another translation unit}}
}

char func_20() {
  return C1<int>::C2<long>::s_tvar_2<char>;
}

char func_21() {
  return C1<int>::C2<long>::s_tvar_2<long>; // expected-warning{{instantiation of variable 'C1<int>::C2<long>::s_tvar_2<long>' required here, but no definition is available}}
                                  // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::C2<long>::s_tvar_2<long>' is explicitly instantiated in another translation unit}}
}

void func_22(C1<int>::C2<long> *x) {
  x->tmeth_2<char>();
}

void func_23(C1<int>::C2<long> *x) {
  x->tmeth_2<int>();    // expected-warning{{instantiation of function 'C1<int>::C2<long>::tmeth_2<int>' required here, but no definition is available}}
                        // expected-note@-1{{add an explicit instantiation declaration to suppress this warning if 'C1<int>::C2<long>::tmeth_2<int>' is explicitly instantiated in another translation unit}}
}

namespace test_24 {
  template <typename T> struct X {
    friend void g(int);
    operator int() { return 0; }
  };
  void h(X<int> x) { g(x); } // no warning for use of 'g' despite the declaration having been instantiated from a template
}

#define INCLUDE
#include "undefined-template.cpp"
void func_25(SystemHeader<char> *x) {
  x->meth();
}

int main() {
  return 0;
}
#else
#pragma clang system_header
template <typename T> struct SystemHeader { T meth(); };
#endif
