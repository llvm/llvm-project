// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

namespace [[deprecated]] {}  // expected-warning {{'deprecated' attribute on anonymous namespace ignored}}

namespace [[deprecated]] N { // expected-note 4{{'N' has been explicitly marked deprecated here}}
  int X;
  int Y = X; // Ok
  int f();
}

int N::f() { // Ok
  return Y; // Ok
}

void f() {
  int Y = N::f(); // expected-warning {{'N' is deprecated}}
  using N::X; // expected-warning {{'N' is deprecated}}
  int Z = X; //Ok
}

void g() {
  using namespace N; // expected-warning {{'N' is deprecated}}
  int Z = Y; // Ok
}

namespace M = N; // expected-warning {{'N' is deprecated}}

// Shouldn't diag:
[[nodiscard, deprecated("")]] int PR37935();

namespace cxx20_concept {
template <typename>
concept C __attribute__((deprecated)) = true; // #C

template <C T>
// expected-warning@-1 {{'C' is deprecated}}
//   expected-note@#C {{'C' has been explicitly marked deprecated here}}
void f();

namespace GH98164 {
template <int>
auto b() = delete; // #b

decltype(b<0>()) x; // #x
// expected-error@#x {{call to deleted function 'b'}}
//   expected-note@#b {{candidate function [with $0 = 0] has been explicitly deleted}}
// expected-error@#x {{function 'b<0>' with deduced return type cannot be used before it is defined}}
//   expected-note@#b {{'b<0>' declared here}}
} // namespace GH98164
} // namespace cxx20_concept
