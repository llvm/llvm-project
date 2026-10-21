// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// expected-no-diagnostics
export module fake_std;

extern "C++" {
namespace std {
  using size_t = decltype(sizeof 0);
  export enum class align_val_t : size_t {};
} // namespace std

export void *operator new(std::size_t, std::align_val_t);

namespace std {
void foo() { align_val_t x{}; }
} // namespace std
} // extern "C++"
