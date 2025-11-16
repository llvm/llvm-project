// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

template<typename T>
void f1();
namespace foo {
  template<typename T>
  void f0();
  template<typename T>
  void f1();
}
template<typename T>
void f0() {}
template<typename T>
void f1() {}
