//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -o %t.out -Wno-deprecated-declarations
// RUN: %t.out

// Checks the relational operators of sycl::multi_ptr: comparisons between two
// multi_ptr instances and comparisons against nullptr in both argument orders.
// Each operator is cross-checked against the corresponding standard function
// object, which also exercises multi_ptr as a key of an ordered container.

#include <sycl/sycl.hpp>

#include <cassert>
#include <functional>

namespace as = sycl::access;

template <typename MultiPtrT>
void checkAgainstStdFunctors(MultiPtrT Lhs, MultiPtrT Rhs) {
  assert((Lhs == Rhs) == std::equal_to<MultiPtrT>()(Lhs, Rhs));
  assert((Lhs != Rhs) == std::not_equal_to<MultiPtrT>()(Lhs, Rhs));
  assert((Lhs < Rhs) == std::less<MultiPtrT>()(Lhs, Rhs));
  assert((Lhs > Rhs) == std::greater<MultiPtrT>()(Lhs, Rhs));
  assert((Lhs <= Rhs) == std::less_equal<MultiPtrT>()(Lhs, Rhs));
  assert((Lhs >= Rhs) == std::greater_equal<MultiPtrT>()(Lhs, Rhs));
}

template <typename MultiPtrT> void checkAgainstNullptr(MultiPtrT Ptr) {
  const MultiPtrT Null;

  assert(Null == nullptr);
  assert(nullptr == Null);
  assert(!(Null != nullptr));
  assert(!(nullptr != Null));

  assert(Ptr != nullptr);
  assert(nullptr != Ptr);
  assert(!(Ptr == nullptr));
  assert(!(nullptr == Ptr));

  // The remaining four operators are only required to be consistent with the
  // corresponding comparison of the underlying pointers.
  assert((Ptr < nullptr) == std::less<MultiPtrT>()(Ptr, MultiPtrT{}));
  assert((nullptr < Ptr) == std::less<MultiPtrT>()(MultiPtrT{}, Ptr));
  assert((Ptr <= nullptr) == std::less_equal<MultiPtrT>()(Ptr, MultiPtrT{}));
  assert((nullptr <= Ptr) == std::less_equal<MultiPtrT>()(MultiPtrT{}, Ptr));
  assert((Ptr > nullptr) == std::greater<MultiPtrT>()(Ptr, MultiPtrT{}));
  assert((nullptr > Ptr) == std::greater<MultiPtrT>()(MultiPtrT{}, Ptr));
  assert((Ptr >= nullptr) == std::greater_equal<MultiPtrT>()(Ptr, MultiPtrT{}));
  assert((nullptr >= Ptr) == std::greater_equal<MultiPtrT>()(MultiPtrT{}, Ptr));
}

template <as::address_space Space, as::decorated Decorated>
void checkSpace(int *Data) {
  using multi_ptr_t = sycl::multi_ptr<int, Space, Decorated>;

  multi_ptr_t First = sycl::address_space_cast<Space, Decorated>(Data);
  multi_ptr_t Second = sycl::address_space_cast<Space, Decorated>(Data + 1);
  multi_ptr_t FirstAgain = sycl::address_space_cast<Space, Decorated>(Data);

  // Equality reflects the underlying pointer value.
  assert(First == FirstAgain);
  assert(!(First != FirstAgain));
  assert(First != Second);
  assert(!(First == Second));

  // Ordering follows the underlying pointer ordering.
  assert(First < Second);
  assert(Second > First);
  assert(First <= Second);
  assert(First <= FirstAgain);
  assert(Second >= First);
  assert(First >= FirstAgain);

  checkAgainstStdFunctors(First, Second);
  checkAgainstStdFunctors(First, FirstAgain);
  checkAgainstNullptr(First);
}

template <as::address_space Space> void checkDecorations(int *Data) {
  checkSpace<Space, as::decorated::yes>(Data);
  checkSpace<Space, as::decorated::no>(Data);
  checkSpace<Space, as::decorated::legacy>(Data);
}

int main() {
  int Data[2] = {0, 1};

  checkDecorations<as::address_space::generic_space>(Data);
  checkDecorations<as::address_space::global_space>(Data);
  checkDecorations<as::address_space::local_space>(Data);
  checkDecorations<as::address_space::private_space>(Data);

  // The void specializations provide the relational operators as well.
  using void_ptr_t =
      sycl::multi_ptr<void, as::address_space::global_space, as::decorated::no>;
  const void_ptr_t NullVoid;
  assert(NullVoid == nullptr);
  assert(nullptr == NullVoid);
  assert(!(NullVoid != nullptr));

  using const_void_ptr_t =
      sycl::multi_ptr<const void, as::address_space::global_space,
                      as::decorated::no>;
  const const_void_ptr_t NullConstVoid;
  assert(NullConstVoid == nullptr);
  assert(!(NullConstVoid != nullptr));

  return 0;
}
