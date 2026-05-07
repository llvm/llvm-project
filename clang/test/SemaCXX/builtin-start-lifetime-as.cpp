// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

struct Trivial {
    int x;
    float y;
};

struct NonTrivial {
    int x;
    ~NonTrivial() {}
};

struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}

void test_valid_cases(void* ptr) {
    // Should complete without any diagnostics.
    __builtin_start_lifetime_as((int*)ptr);
    __builtin_start_lifetime_as((Trivial*)ptr);

    // P2679R2 check
    __builtin_start_lifetime_as((int(*)[5])ptr);
  __builtin_start_lifetime_as((Trivial(*)[5][10])ptr);
}

void test_invalid_types(void* ptr) {
  // expected-error@+1 {{type 'NonTrivial' is not an implicit-lifetime type, cannot start lifetime}}
  __builtin_start_lifetime_as((NonTrivial*)ptr);

  // expected-error@+1 {{type 'NonTrivial[5]' is not an implicit-lifetime type, cannot start lifetime}}
  __builtin_start_lifetime_as((NonTrivial(*)[5])ptr);

  // expected-error@+1 {{type 'void' is not an implicit-lifetime type, cannot start lifetime}}
  __builtin_start_lifetime_as((void*)ptr);

  // expected-error@+1 {{type 'void ()' is not an implicit-lifetime type, cannot start lifetime}}
  __builtin_start_lifetime_as((void(*)())ptr);
}

void test_incomplete_types(void* ptr) {
  // expected-error@+1 {{incomplete type 'Incomplete' where a complete type is required}}
  __builtin_start_lifetime_as((Incomplete*)ptr);
}

void test_invalid_arguments() {
  int x;
  // expected-error@+1 {{passing 'int' to parameter of incompatible type pointer}}
  __builtin_start_lifetime_as(x);
  
  // expected-error@+1 {{too few arguments to function call, expected 1, have 0}}
  __builtin_start_lifetime_as();
}

void test_constexpr() {
  // expected-error@+2 {{constexpr variable 'ptr' must be initialized by a constant expression}}
  constexpr auto* ptr = (int*)__builtin_start_lifetime_as((int*)nullptr);
}