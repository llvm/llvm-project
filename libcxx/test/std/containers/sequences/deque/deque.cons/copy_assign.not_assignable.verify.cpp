//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Make sure that a std::deque containing move-only types can't be copy-assigned.

#include <deque>

struct AttributeType {
  AttributeType();
  AttributeType(const AttributeType&);
  AttributeType(AttributeType&&);
  AttributeType& operator=(const AttributeType&) = delete;
  AttributeType& operator=(AttributeType&&);
  ~AttributeType();
};

void f() {
  std::deque<AttributeType> v;
  std::deque<AttributeType> copy;
  copy = v;
  // expected-error@* {{T must be Cpp17CopyAssignable}}
  // expected-error@* {{overload resolution selected deleted operator '='}}
}
