//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right, class AccessorPolicy = default_accessor>
// class mdspan;
//
// Mandates:
//   - ElementType is a complete object type that is neither an abstract class type nor an array type.
//   - is_same_v<ElementType, typename AccessorPolicy::element_type> is true.

#include <mdspan>

struct Incomplete;

class AbstractClass {
public:
  virtual void method() = 0;
};

struct VoidAccessor {
  using offset_policy    = VoidAccessor;
  using element_type     = void;
  using reference        = void;
  using data_handle_type = element_type*;
  reference access(data_handle_type, std::size_t) const;
};

struct RefAccessor {
  using offset_policy    = RefAccessor;
  using element_type     = int&;
  using reference        = int&;
  using data_handle_type = int*;
  reference access(data_handle_type p, std::size_t i) const { return p[i]; }
};

struct FuncAccessor {
  using offset_policy    = FuncAccessor;
  using element_type     = int();
  using reference        = int (&)();
  using data_handle_type = int (*)();
  reference access(data_handle_type, std::size_t) const;
};

void incomplete_object_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: ElementType template parameter must be a complete object type}}
  [[maybe_unused]] std::mdspan<Incomplete, std::dextents<std::size_t, 2>> m;
}

void void_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: ElementType template parameter must be a complete object type}}
  [[maybe_unused]] std::mdspan<void, std::dextents<std::size_t, 2>, std::layout_right, VoidAccessor> m;
}

void reference_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: ElementType template parameter must be a complete object type}}
  [[maybe_unused]] std::mdspan<int&, std::dextents<std::size_t, 2>, std::layout_right, RefAccessor> m;
}

void function_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: ElementType template parameter must be a complete object type}}
  [[maybe_unused]] std::mdspan<int(), std::dextents<std::size_t, 2>, std::layout_right, FuncAccessor> m;
}

void not_abstract_class() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: ElementType template parameter may not be an abstract class}}
  [[maybe_unused]] std::mdspan<AbstractClass, std::extents<int>> m;
}

void not_array_type() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: ElementType template parameter may not be an array type}}
  [[maybe_unused]] std::mdspan<int[5], std::extents<int>> m;
}

void element_type_mismatch() {
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: ElementType template parameter must match AccessorPolicy::element_type}}
  [[maybe_unused]] std::mdspan<int, std::extents<int>, std::layout_right, std::default_accessor<const int>> m;
}
