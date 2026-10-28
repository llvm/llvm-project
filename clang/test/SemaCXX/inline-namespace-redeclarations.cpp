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
} // namespace v1
namespace ns {
static_assert(value == 1,
              "'value' should be found in the parent scope of 'ns', which is 'v1'");
} // namespace ns
} // namespace reopen

namespace qualified {
constexpr int value = 0;
inline namespace v1 {
constexpr int value = 1;
struct X;
enum class E;
} // namespace v1
} // namespace qualified
namespace alias = qualified;
struct alias::X {
  static_assert(value == 1,
                "'value' should be found in the parent scope of 'X', which is 'v1'");
};
enum class alias::E { test = value };
static_assert(static_cast<int>(alias::E::test) == 1,
              "'value' was supposed to be found in the parent scope of 'E', which is 'v1'");
