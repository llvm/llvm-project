// RUN: %clang_cc1 -triple x86_64 -fcxx-exceptions -std=c++20 -fblocks -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -triple x86_64 -fcxx-exceptions -std=c++20 -fblocks                                         -verify=ref,both %s


struct S {
  void (^p)(){}; // expected-note {{invalid type 'void (^)()' is a member of 'S'}}
};
constexpr long l = __builtin_bit_cast(long, S{}); // both-error {{must be initialized by a constant expression}} \
                                                  // both-note {{constexpr bit cast involving type 'void (^)()' is not yet supported}}
