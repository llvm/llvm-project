// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.NullTerminated -verify %s

#define NULL_TERMINATED __attribute__((annotate("null_terminated")))

void receive(NULL_TERMINATED const int signals[]);

void test_constexpr_bad() {
  constexpr int sigs[] = {1, 2, 3};
  receive(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_constexpr_good() {
  constexpr int sigs[] = {1, 2, 0};
  receive(sigs);
}

void test_constexpr_early_term() {
  constexpr int sigs[] = {1, 0, 3};
  receive(sigs);
}

void receive_ref(NULL_TERMINATED const int (&signals)[3]);

void test_ref_bad() {
  const int sigs[3] = {1, 2, 3};
  receive_ref(sigs);  // expected-warning{{array argument is not null-terminated}}
}

void test_ref_good() {
  const int sigs[3] = {1, 2, 0};
  receive_ref(sigs);
}

// CSA limitation - CSA cannot see through default member initializers.
struct S {
  int bad[3] = {1, 2, 3};
  int good[3] = {1, 2, 0};

};
void test_inclass_bad() {
  S s;
  receive(s.bad);
}

void test_inclass_good() {
  S s;
  receive(s.good);
}

// Test C++11 attribute syntax
void receive_cpp([[clang::annotate("null_terminated")]] const int signals[]);

void test_cpp_spelling() {
  int sigs[] = {1, 2, 3};
  receive_cpp(sigs);  // expected-warning{{array argument is not null-terminated}}
}
