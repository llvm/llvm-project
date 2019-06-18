// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// template <class T>
// variant& operator=(T&&) noexcept(see below);

#include <variant>
#include <string>
#include <memory>

int main(int, char**)
{
  std::variant<int, int> v1;
  std::variant<long, long long> v2;
  std::variant<char> v3;
  v1 = 1; // expected-error {{no viable overloaded '='}}
  v2 = 1; // expected-error {{no viable overloaded '='}}
  v3 = 1; // expected-error {{no viable overloaded '='}}

  std::variant<std::string, float> v4;
  std::variant<std::string, double> v5;
  std::variant<std::string, bool> v6;
  v4 = 1; // expected-error {{no viable overloaded '='}}
  v5 = 1; // expected-error {{no viable overloaded '='}}
  v6 = 1; // expected-error {{no viable overloaded '='}}

  std::variant<int, bool> v7;
  std::variant<int, bool const> v8;
  std::variant<int, bool volatile> v9;
  v7 = "meow"; // expected-error {{no viable overloaded '='}}
  v8 = "meow"; // expected-error {{no viable overloaded '='}}
  v9 = "meow"; // expected-error {{no viable overloaded '='}}

  std::variant<bool> v10;
  std::variant<bool> v11;
  std::variant<bool> v12;
  v10 = std::true_type(); // expected-error {{no viable overloaded '='}}
  v11 = std::unique_ptr<char>(); // expected-error {{no viable overloaded '='}}
  v12 = nullptr; // expected-error {{no viable overloaded '='}}
}
