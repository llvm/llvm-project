// RUN: %clang_cc1 %s -verify -fsyntax-only

template <class T>
struct [[clang::complete_on_member_instantiation]] S1 {}; // expected-error {{'clang::complete_on_member_instantiation' attribute takes one argument}}

template <class T>
struct [[clang::complete_on_member_instantiation(1)]] S2 {}; // expected-error {{expected a type}}

template <class T>
struct [[clang::complete_on_member_instantiation(T)]] RequiresComplete {
  void func() {}
};

struct Complete {};
struct Incomplete; // expected-note 7 {{forward declaration of 'Incomplete'}}

void func(RequiresComplete<Incomplete>& incomplete, RequiresComplete<Complete>& complete) {
  complete.func();
  incomplete.func(); // expected-error {{'Incomplete' has to be complete when calling a member function}}
}

template <class T, class U>
struct [[clang::complete_on_member_instantiation(T), clang::complete_on_member_instantiation(U)]] RequiresComplete2 {
  void func() {}
};

void func2(RequiresComplete2<Incomplete, Incomplete> both_incomplete,
           RequiresComplete2<Incomplete, Complete> first_incomplete,
           RequiresComplete2<Complete, Incomplete> second_incomplete) {
  both_incomplete.func();   // expected-error {{'Incomplete' has to be complete when calling a member function}}
  first_incomplete.func();  // expected-error {{'Incomplete' has to be complete when calling a member function}}
  second_incomplete.func(); // expected-error {{'Incomplete' has to be complete when calling a member function}}
}

template <class T, class U>
struct [[clang::complete_on_member_instantiation(T), clang::complete_on_member_instantiation(U)]] RequiresComplete3;

template <class T, class U>
struct RequiresComplete3 {
  void func() {}
};

void func2(RequiresComplete3<Incomplete, Incomplete> both_incomplete,
           RequiresComplete3<Incomplete, Complete> first_incomplete,
           RequiresComplete3<Complete, Incomplete> second_incomplete) {
  both_incomplete.func();   // expected-error {{'Incomplete' has to be complete when calling a member function}}
  first_incomplete.func();  // expected-error {{'Incomplete' has to be complete when calling a member function}}
  second_incomplete.func(); // expected-error {{'Incomplete' has to be complete when calling a member function}}
}
