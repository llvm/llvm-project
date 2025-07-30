//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>

// template<class ElementType, size_t ByteAlignment>
// class aligned_accessor;

// ElementType is required to be a complete object type that is neither an abstract class type nor an array type.

#include <mdspan>

class AbstractClass {
public:
  virtual void method() = 0;
};

void not_abstract_class() {
  // expected-error-re@*:* {{static assertion failed {{.*}}aligned_accessor: template argument may not be an abstract class}}
  [[maybe_unused]] std::aligned_accessor<AbstractClass, 8> acc;
}

void not_array_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}aligned_accessor: template argument may not be an array type}}
  [[maybe_unused]] std::aligned_accessor<int[5], 128> acc;
}
