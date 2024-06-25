// RUN: %clang_cc1 -std=c++20 %s -Wall -fsyntax-only -verify

// expected-no-diagnostics
export module a;

export constexpr auto a = []{};
