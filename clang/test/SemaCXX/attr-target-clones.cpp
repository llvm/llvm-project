// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14

// expected-error@+2 {{attribute 'target_clones' multiversioned functions do not yet support function templates}}
template<typename T, typename U>
int __attribute__((target_clones("sse4.2", "default"))) foo(){ return 1;}

void uses_lambda() {
  // expected-error@+1 {{attribute 'target_clones' multiversioned functions do not yet support lambdas}}
  auto x = []()__attribute__((target_clones("sse4.2", "arch=ivybridge", "default"))) {};
  x();
}
