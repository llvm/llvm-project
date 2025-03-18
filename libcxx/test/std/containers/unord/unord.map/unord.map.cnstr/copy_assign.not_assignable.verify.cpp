//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Make sure that a std::unordered_map containing move-only types can't be copy-assigned.

#include <unordered_map>

struct AttributeType {
  AttributeType();
  AttributeType(const AttributeType&);
  AttributeType(AttributeType&&);
  AttributeType& operator=(const AttributeType&) = delete;
  AttributeType& operator=(AttributeType&&);
  ~AttributeType();
  bool operator==(const AttributeType&) const;
};

template <>
struct std::hash<AttributeType> {
  size_t operator()(const AttributeType&) const;
};

void f() {
  std::unordered_map<int, AttributeType> v;
  std::unordered_map<int, AttributeType> copy;
  copy = v;
  // expected-error@* {{T must be Cpp17CopyAssignable}}
  // expected-error@* {{no viable overloaded '='}}
}
