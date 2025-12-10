//RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// Force a fatal error
#include <unknownheader> // expected-error {{file not found}}

namespace N {}

enum class E {};
using enum N::E::V;

using T = int;
using enum N::T::v;
