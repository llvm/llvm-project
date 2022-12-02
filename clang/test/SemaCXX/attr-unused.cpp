// RUN: %clang_cc1 -verify -Wunused -Wused-but-marked-unused -fsyntax-only %s

namespace ns_unused { typedef int Int_unused __attribute__((unused)); }
namespace ns_not_unused { typedef int Int_not_unused; }

template <typename T> class C;
template <> class __attribute__((unused)) C<int> {};

void f() {
  ns_not_unused::Int_not_unused i1; // expected-warning {{unused variable}}
  ns_unused::Int_unused i0; // expected-warning {{'Int_unused' was marked unused but was used}}

  union __attribute__((unused)) { // expected-warning {{was marked unused but was used}}
    int i;
  };
  (void) i;

  C<int>(); // expected-warning {{'C<int>' was marked unused but was used}}
}
