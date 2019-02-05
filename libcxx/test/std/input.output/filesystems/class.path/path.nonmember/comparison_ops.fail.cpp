//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <filesystem>


#include "filesystem_include.hpp"

using namespace fs;

struct ConvToPath {
  operator fs::path() const {
    return "";
  }
};

int main() {
  ConvToPath LHS, RHS;
  (void)(LHS == RHS); // expected-error {{invalid operands to binary expression}}
  (void)(LHS != RHS); // expected-error {{invalid operands to binary expression}}
  (void)(LHS < RHS); // expected-error {{invalid operands to binary expression}}
  (void)(LHS <= RHS); // expected-error {{invalid operands to binary expression}}
  (void)(LHS > RHS); // expected-error {{invalid operands to binary expression}}
  (void)(LHS >= RHS); // expected-error {{invalid operands to binary expression}}
}