//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <type_traits>

// has_unique_object_representations

// Verify that has_unique_object_representations(_v) rejects incomplete class and enumeration types and arrays thereof.

#include <type_traits>

class IC;

constexpr bool v1 = std::has_unique_object_representations<IC>::value;
constexpr bool v2 = std::has_unique_object_representations<IC[]>::value;
constexpr bool v3 = std::has_unique_object_representations<IC[1]>::value;
constexpr bool v4 = std::has_unique_object_representations<IC[][1]>::value;

constexpr bool v5 = std::has_unique_object_representations_v<IC>;
constexpr bool v6 = std::has_unique_object_representations_v<IC[]>;
constexpr bool v7 = std::has_unique_object_representations_v<IC[1]>;
constexpr bool v8 = std::has_unique_object_representations_v<IC[][1]>;

// expected-error@*:* 8 {{incomplete type 'IC' used in type trait expression}}

enum E {
  v9  = std::has_unique_object_representations<E>::value,
  v10 = std::has_unique_object_representations<E[]>::value,
  v11 = std::has_unique_object_representations<E[1]>::value,
  v12 = std::has_unique_object_representations<E[][1]>::value,

  v13 = std::has_unique_object_representations_v<E>,
  v14 = std::has_unique_object_representations_v<E[]>,
  v15 = std::has_unique_object_representations_v<E[1]>,
  v16 = std::has_unique_object_representations_v<E[][1]>,

// TODO: Remove the guard once https://llvm.org/PR169472 is resolved.
#ifndef _MSC_VER // In Clang-cl mode, E is incorrectly considered complete here.
// expected-error@*:* 8 {{incomplete type 'E' used in type trait expression}}
#endif
};
