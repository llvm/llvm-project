//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Make sure that a std::vector containing move-only types can't be copy-assigned.

#include <vector>

struct AttributeType {
  AttributeType();
  AttributeType(const AttributeType&);
  AttributeType(AttributeType&&);
  AttributeType& operator=(const AttributeType&) = delete;
  AttributeType& operator=(AttributeType&&);
  ~AttributeType();
};

void f() {
  std::vector<AttributeType> v;
  std::vector<AttributeType> copy;
  copy = v;
  // expected-error@* {{T must be Cpp17CopyAssignable}}
  // expected-error-re@* {{{{no matching function for call to object of type 'const __copy_n'|overload resolution selected deleted operator '='}}}}
}
