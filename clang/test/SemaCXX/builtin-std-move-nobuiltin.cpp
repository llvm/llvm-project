// RUN: %clang_cc1 -std=c++20 -verify %s -DBUILTIN=builtin
// RUN: %clang_cc1 -std=c++20 -verify %s -DBUILTIN=nobuiltin -fno-builtin
// RUN: %clang_cc1 -std=c++20 -verify %s -DBUILTIN=nobuiltin -fno-builtin-std-move -fno-builtin-std-move_if_noexcept -fno-builtin-std-forward
// RUN: %clang_cc1 -std=c++20 -verify %s -DBUILTIN=nobuiltin -ffreestanding
// expected-no-diagnostics

int nobuiltin;

namespace std {
  template<typename T> constexpr T &&move(T &x) { return (T&&)nobuiltin; }
  template<typename T> constexpr T &&move_if_noexcept(T &x) { return (T&&)nobuiltin; }
  template<typename T> constexpr T &&forward(T &x) { return (T&&)nobuiltin; }
}

template<typename T> constexpr T *addr(T &&r) { return &r; }

int builtin;
static_assert(addr(std::move(builtin)) == &BUILTIN);
static_assert(addr(std::move_if_noexcept(builtin)) == &BUILTIN);
static_assert(addr(std::forward(builtin)) == &BUILTIN);
