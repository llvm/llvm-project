//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// A program that instantiates the definition of unexpected for a non-object type, an array type, a specialization of unexpected, or a cv-qualified type is ill-formed.

#include <expected>



template class std::unexpected<int[2]>; // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}[expected.un.general]}}

template class std::unexpected<const int>; // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}[expected.un.general]}}

template class std::unexpected<int&>; // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}[expected.un.general]}}

template class std::unexpected<std::unexpected<int>>; // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}[expected.un.general]}}

template class std::unexpected<volatile int>; // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}[expected.un.general]}}
