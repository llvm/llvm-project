//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_OPERATOR_HIJACKER_H
#define SUPPORT_OPERATOR_HIJACKER_H

#include <cstddef>
#include <functional>

#include "test_macros.h"

/// Helper struct to test ADL-hijacking in containers.
///
/// The class has some additional operations to be usable in all containers.
struct operator_hijacker {
  bool operator<(const operator_hijacker&) const { return true; }
  bool operator==(const operator_hijacker&) const { return true; }

  template <typename T>
  friend void operator&(T&&) = delete;
  template <class T, class U>
  friend void operator,(T&&, U&&) = delete;
  template <class T, class U>
  friend void operator&&(T&&, U&&) = delete;
  template <class T, class U>
  friend void operator||(T&&, U&&) = delete;
};

static_assert(std::is_trivially_copyable<operator_hijacker>::value &&     //
                  std::is_copy_constructible<operator_hijacker>::value && //
                  std::is_move_constructible<operator_hijacker>::value && //
                  std::is_copy_assignable<operator_hijacker>::value &&    //
                  std::is_move_assignable<operator_hijacker>::value,      //
              "does not satisfy the requirements for atomic<operator_hijacker>");

template <>
struct std::hash<operator_hijacker> {
  std::size_t operator()(const operator_hijacker&) const { return 0; }
};

#endif // SUPPORT_OPERATOR_HIJACKER_H
