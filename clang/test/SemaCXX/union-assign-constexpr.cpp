// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fsyntax-only -verify %s
// expected-no-diagnostics

// The memcpy body must not block constant evaluation of a union assignment.

union U {
  int a;
  float b;
};

constexpr int copy_active() {
  U x{};
  x.a = 7;
  U y{};
  y = x;
  return y.a;
}

constexpr int move_active() {
  U x{};
  x.a = 9;
  U y{};
  y = static_cast<U &&>(x);
  return y.a;
}

static_assert(copy_active() == 7);
static_assert(move_active() == 9);
