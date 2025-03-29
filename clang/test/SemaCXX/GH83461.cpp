// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

struct S {
  template<typename Ty = int>
  friend void foo(auto){}

  template<typename Ty = int, typename Tz>
  friend void foo2(){}
};

template<typename T>
struct TemplS {
  template<typename Ty = int>
  friend void foo3(auto){}

  template<typename Ty = int, typename Tz>
  friend void foo4(){}
};

void Inst() {
  TemplS<int>();
}
// expected-error@+2{{template parameter missing a default argument}}
// expected-note@+1{{previous default template argument defined here}}
template<typename T = int, typename U>
struct ClassTempl{};

struct HasFriendClassTempl {
  // expected-error@+1{{default template argument not permitted on a friend template}}
  template<typename T = int, typename U>
  friend struct Friend;

  // expected-error@+3{{cannot define a type in a friend declaration}}
  // expected-error@+1{{default template argument not permitted on a friend template}}
  template<typename T = int, typename U>
  friend struct Friend2{};
};

template<typename Ty>
struct HasFriendClassTempl2 {
  // expected-error@+3{{template parameter missing a default argument}}
  // expected-note@+2{{previous default template argument defined here}}
  // expected-note@#INST2{{in instantiation of template class}}
  template<typename T = int, typename U>
  friend struct Friend;
};

void Inst2() {
  HasFriendClassTempl2<int>(); // #INST2
}

// expected-error@+2{{template parameter missing a default argument}}
// expected-note@+1{{previous default template argument defined here}}
template<typename T = int, typename U>
static constexpr U VarTempl;

// expected-error@+2{{template parameter missing a default argument}}
// expected-note@+1{{previous default template argument defined here}}
template<typename T = int, typename U>
using TypeAlias = U;
