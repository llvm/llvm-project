// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++11 -verify %s

static_assert(true, "");
static_assert(false, ""); // expected-error{{failed}}
static_assert(nullptr == nullptr, "");
static_assert(1 == 1, "");
static_assert(1 == 3, ""); // expected-error{{failed}}

constexpr int number = 10;
static_assert(number == 10, "");
static_assert(number != 10, ""); // expected-error{{failed}}

constexpr bool getTrue() { return true; }
constexpr bool getFalse() { return false; }
constexpr void* getNull() { return nullptr; }
