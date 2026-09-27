// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// The constant evaluator used to assume the (std::nothrow) placement argument
// is always an lvalue and crashed when a user-declared allocation function
// takes std::nothrow_t by value, making the argument a prvalue.

namespace std {
inline constexpr struct nothrow_t {
} nothrow;
} // namespace std

void *operator new[](unsigned long, std::nothrow_t) noexcept;
void *operator new(unsigned long, std::nothrow_t) noexcept;

void set(int *p) {
  p = (1 ? new (std::nothrow) int[1] : nullptr);
  p = (1 ? new (std::nothrow) int : nullptr);
}
