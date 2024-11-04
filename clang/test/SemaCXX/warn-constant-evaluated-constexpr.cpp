// RUN: %clang_cc1 -std=c++2a -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++2a -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

namespace std {
constexpr bool is_constant_evaluated() noexcept {
  return __builtin_is_constant_evaluated();
}
} // namespace std

constexpr int fn1() {
  if constexpr (std::is_constant_evaluated()) // expected-warning {{'std::is_constant_evaluated' will always evaluate to 'true' in a manifestly constant-evaluated expression}}
    return 0;
  else
    return 1;
}

constexpr int fn2() {
  if constexpr (!std::is_constant_evaluated()) // expected-warning {{'std::is_constant_evaluated' will always evaluate to 'true' in a manifestly constant-evaluated expression}}
    return 0;
  else
    return 1;
}

constexpr int fn3() {
  if constexpr (std::is_constant_evaluated() == false) // expected-warning {{'std::is_constant_evaluated' will always evaluate to 'true' in a manifestly constant-evaluated expression}}
    return 0;
  else
    return 1;
}

constexpr int fn4() {
  if constexpr (__builtin_is_constant_evaluated() == true) // expected-warning {{'__builtin_is_constant_evaluated' will always evaluate to 'true' in a manifestly constant-evaluated expression}}
    return 0;
  else
    return 1;
}

constexpr int fn5() {
  if constexpr (__builtin_is_constant_evaluated()) // expected-warning {{'__builtin_is_constant_evaluated' will always evaluate to 'true' in a manifestly constant-evaluated expression}}
    return 0;
  else
    return 1;
}

constexpr int nowarn1() {
  if (std::is_constant_evaluated())
    return 0;
  else
    return 1;
}

constexpr int nowarn2() {
  if (!__builtin_is_constant_evaluated())
    return 0;
  else
    return 1;
}
