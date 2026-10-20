// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

// Lookup must start in the semantic parent, rather than making the names in
// the inline namespace ambiguous with those in its enclosing namespace.
namespace reopen {
constexpr int value = 0;
inline namespace v1 {
constexpr int value = 1;
namespace ns {}
}
namespace ns {
static_assert(value == 1, "lookup in the inline namespace");
}
}

namespace qualified {
constexpr int value = 0;
inline namespace v1 {
constexpr int value = 1;
struct X;
enum class E;
}
}
namespace alias = qualified;
struct alias::X {
  static_assert(value == 1, "lookup in the inline namespace");
};
enum class alias::E { test = value };
static_assert(static_cast<int>(alias::E::test) == 1,
              "enum definitions retain their namespace too");
