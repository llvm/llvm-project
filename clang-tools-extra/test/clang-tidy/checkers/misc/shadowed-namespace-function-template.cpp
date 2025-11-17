// RUN: %check_clang_tidy %s misc-shadowed-namespace-function %t

template<typename T>
void f1();
namespace foo {
  template<typename T>
  void f0();
  template<typename T>
  void f1();
}

// FIXME: provide warning in these two cases
// FIXME: provide fixit for f0
template<typename T>
void f0() {}
template<typename T>
void f1() {}
