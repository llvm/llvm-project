//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that functions are marked [[nodiscard]] as an extension in C++17

// [[nodiscard]] std::make_unique(Args&&...);
// [[nodiscard]] std::make_unique(size_t);
// [[nodiscard]] std::make_unique_for_overwrite();
// [[nodiscard]] std::make_unique_for_overwrite(size_t);

// [[nodiscard]] std::make_shared(Args&&...);
// [[nodiscard]] std::make_shared(size_t);
// [[nodiscard]] std::make_shared_for_overwrite();
// [[nodiscard]] std::make_shared_for_overwrite(size_t);
// [[nodiscard]] std::make_shared(size_t, const remove_extent_t<_Tp>&);
// [[nodiscard]] std::make_shared(const remove_extent_t<_Tp>&);

#include <memory>

void f() {

    std::make_unique<int>(1); // expected-warning {{ignoring return value of function}}
    std::make_unique<int[]>(5); // expected-warning {{ignoring return value of function}}
    std::make_unique_for_overwrite<int>(); // expected-warning {{ignoring return value of function}}
    std::make_unique_for_overwrite<int[]>(5); // expected-warning {{ignoring return value of function}}

    std::make_shared<int>(1); // expected-warning {{ignoring return value of function}}
    std::make_shared<int[]>(5); // expected-warning {{ignoring return value of function}}
    std::make_shared_for_overwrite<int>(); // expected-warning {{ignoring return value of function}}
    std::make_shared_for_overwrite<int[]>(5); // expected-warning {{ignoring return value of function}}
    std::make_shared<int[]>(5, 1); // expected-warning {{ignoring return value of function}}
    std::make_shared<int[]>(1); // expected-warning {{ignoring return value of function}}

}
