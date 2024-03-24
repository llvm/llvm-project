// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-c++23-extensions
// RUN: %clang_cc1 -fsyntax-only -verify %s -ffreestanding -Wno-c++23-extensions

[[clang::builtin]] void func(); // expected-error {{'builtin' attribute takes one argument}}
[[clang::builtin("unknown_builtin")]] void func(); // expected-warning {{builtin is not supported}}
[[clang::builtin("memcpy")]] void func(); // expected-error {{function signature does not match the signature of the builtin}} \
                                             expected-note {{expected signature is 'void *(void *, const void *, unsigned long)'}}
[[clang::builtin("move")]] void func(); // expected-warning {{builtin is not supported}}

// has unevaluated parameters
[[clang::builtin("__builtin_constant_p")]] void constant_p(); // expected-warning {{builtin is not supported}}

[[clang::builtin("memcpy")]] void* my_memcpy(void*, const void*, unsigned long);
[[clang::builtin("memcpy")]] char* my_memcpy(char*, const char*, unsigned long);
[[clang::builtin("memcpy")]] char* my_memcpy(char*, const char*); // expected-error {{function signature does not match the signature of the builtin}} \
                                                                     expected-note {{expected signature is}}

[[clang::builtin("memmove")]] char* typed_memmove(char*, const char*, unsigned long); // expected-note {{candidate function}}

void call_memmove(void* ptr) {
  typed_memmove(ptr, ptr, 1); // expected-error {{no matching function for call to 'typed_memmove'}}
}

[[clang::builtin("__builtin_memmove")]] void* non_constexpr_memmove(void*, const void*, unsigned long);

constexpr void call_non_constexpr_memmove() { // expected-error {{constexpr function never produces a constant expression}}
  int i = 0;
  int j = 0;
  non_constexpr_memmove(&j, &i, sizeof(int)); // expected-note {{subexpression not valid in a constant expression}}
}

[[clang::builtin("__builtin_memmove")]] constexpr void* constexpr_memmove(void*, const void*, unsigned long);

constexpr void call_constexpr_memmove() {
  int i = 0;
  int j = 0;
  constexpr_memmove(&j, &i, sizeof(int));
}

// allows type mismatches
[[clang::builtin("std::move")]] void my_move(); // expected-error {{expected 1 argument but got 0}}
[[clang::builtin("std::move")]] void my_move(int);

void call_move() {
  my_move(1); // expected-error {{unsupported signature for 'my_move'}}
}

// has custom type checking
[[clang::builtin("__builtin_operator_new")]] void* my_operator_new(unsigned long); // expected-warning {{builtin is not supported}}

// canonical types are compared
using size_t = decltype(sizeof(int));
[[clang::builtin("__builtin_memcmp")]] int my_memcmp(const char*, const char*, size_t);

struct reject_on_member_functions {
  template<class T>
  [[clang::builtin("std::forward")]] T&& operator()(T&&) noexcept; // expected-warning {{attribute 'builtin' is not supported on member functions}}
  [[clang::builtin("memchr")]] const void* memchr(const void*, int, unsigned long); // expected-warning {{attribute 'builtin' is not supported on member functions}}
};

struct accept_on_static_member_functions {
  template <class T>
  [[clang::builtin("std::forward")]] static T&& operator()(T&&) noexcept;
  [[clang::builtin("memchr")]] static const void* memchr(const void*, int, unsigned long);
};
