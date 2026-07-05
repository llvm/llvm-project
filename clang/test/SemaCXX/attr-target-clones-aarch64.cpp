// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14

void lambda() {
  // expected-error@+1 {{attribute 'target_clones' multiversioned functions do not yet support lambdas}}
  auto x = []() __attribute__((target_clones("default"))){};
  x();
  // expected-error@+1 {{attribute 'target_clones' multiversioned functions do not yet support lambdas}}
  auto y = []() __attribute__((target_clones("fp16+lse", "rdm"))){};
  y();
}
