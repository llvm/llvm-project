// RUN: %clang_cc1 -std=c++11 %s -Wunused -Wno-unused-lambda-capture -verify

struct MoveOnly {
  MoveOnly(MoveOnly&&);
  MoveOnly(const MoveOnly&);
};

template<typename T> T &&move(T&);
void test_special_member_functions(MoveOnly mo, int i) {
  auto lambda1 = [i]() { }; // expected-note 2{{lambda expression begins here}} expected-note 2{{candidate}}

  // Default constructor
  decltype(lambda1) lambda2; // expected-error{{no matching constructor}}

  // Copy assignment operator
  lambda1 = lambda1; // expected-error{{copy assignment operator is implicitly deleted}}

  // Move assignment operator
  lambda1 = move(lambda1); // expected-error{{copy assignment operator is implicitly deleted}}

  // Copy constructor
  decltype(lambda1) lambda3 = lambda1;
  decltype(lambda1) lambda4(lambda1);

  // Move constructor
  decltype(lambda1) lambda5 = move(lambda1);
  decltype(lambda1) lambda6(move(lambda1));
}
