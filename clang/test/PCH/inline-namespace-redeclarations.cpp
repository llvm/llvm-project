// RUN: %clang_cc1 -std=c++11 -include %s -verify %s
// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t -verify %s

#ifndef HEADER
#define HEADER

namespace reopen {
constexpr int value = 0;
inline namespace v1 {
constexpr int value = 1;
namespace ns {
struct Y;
} // namespace ns
} // namespace v1
namespace ns {
struct Y {};
} // namespace ns
} // namespace reopen

namespace qualified {
constexpr int value = 0;
inline namespace v1 {
constexpr int value = 1;
struct X;
} // namespace v1
} // namespace qualified
namespace alias = qualified;
struct alias::X {
  static constexpr int get();
};

#else

// The previous definitions may have been read from the PCH.
// expected-note@* 2 {{previous definition is here}}

namespace reopen {
namespace ns {
static_assert(
    value == 1,
    "'value' should be found in the parent scope of 'ns', which is 'v1'");
struct Y {}; // expected-error {{redefinition of 'Y'}}
} // namespace ns
} // namespace reopen

constexpr int alias::X::get() { return value; }
static_assert(
    alias::X::get() == 1,
    "'value' should be found in the parent scope of 'X', which is 'v1'");
struct alias::X {}; // expected-error {{redefinition of 'X'}}

#endif
