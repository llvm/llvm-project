// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14

template <typename T>
T __attribute__((target_clones("sse4.2", "default"))) templated(T value) {
  return value;
}

int use_templated() {
  int (*ptr)(int) = &templated<int>;
  return ptr(1) + templated(2.0f);
}

template double templated<double>(double);

struct HasMemberTemplate {
  template <typename T>
  T __attribute__((target_clones("sse4.2", "default"))) member(T value) {
    return value;
  }
};

int use_member_template(HasMemberTemplate &object) {
  return object.member(1);
}

// expected-error@+2 {{attribute 'target_clones' multiversioned functions do not yet support deduced return types}}
template <typename T>
auto __attribute__((target_clones("sse4.2", "default"))) undeduced(T);

void uses_lambda() {
  // expected-error@+1 {{attribute 'target_clones' multiversioned functions do not yet support lambdas}}
  auto x = []()__attribute__((target_clones("sse4.2", "arch=ivybridge", "default"))) {};
  x();
}
