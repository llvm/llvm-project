// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected -std=c++11 -triple x86_64-linux -pedantic %s
// RUN: %clang_cc1 -verify=both,ref -std=c++11 -triple x86_64-linux -pedantic %s

struct T { int n; };
const T t = { 42 }; // both-note 2{{declared here}}
struct S {
  int m : t.n; // both-warning {{width of bit-field 'm' (42 bits)}} \
               // both-warning {{expression is not an integral constant expression}} \
               // both-note {{read of non-constexpr variable 't' is not allowed}}
};

static_assert(t.n == 42, ""); // both-error {{expression is not an integral constant expression}} \
                              // both-note {{read of non-constexpr variable 't' is not allowed}}
