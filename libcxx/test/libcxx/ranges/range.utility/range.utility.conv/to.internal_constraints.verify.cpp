//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that the "mandates" requirements on the given container are checked correctly using `static_assert`.

#include <ranges>
#include <vector>

template <bool HasDefaultCtr = true, bool HasSingleArgCtr = true,
          bool HasInsert = true, bool HasInsertWithRightSignature = true,
          bool HasPushBack = true, bool HasPushBackWithRightSignature = true>
struct Container {
  using value_type = int;

  int* begin() const { return nullptr; }
  int* end() const { return nullptr; }

  Container()
  requires HasDefaultCtr = default;

  Container(int)
  requires HasSingleArgCtr {
  }

  int* insert(int*, int)
  requires (HasInsert && HasInsertWithRightSignature) {
    return nullptr;
  }

  int* insert()
  requires (HasInsert && !HasInsertWithRightSignature) {
    return nullptr;
  }

  void push_back(int)
  requires (HasPushBack && HasPushBackWithRightSignature) {
  }

  void push_back()
  requires (HasPushBack && !HasPushBackWithRightSignature) {
  }

};

void test() {
  using R = std::vector<int>;
  R in = {1, 2, 3};

  // Case 4 -- default-construct (or construct from the extra arguments) and insert.
  { // All constraints satisfied.
    using C = Container<>;
    (void)std::ranges::to<C>(in);
    (void)std::ranges::to<C>(in, 1);
    (void)std::ranges::to<C>(in, 1.0);
  }

  { // No default constructor.
    using C = Container</*HasDefaultCtr=*/false>;
    (void)std::ranges::to<C>(in); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}ranges::to: unable to convert to the given container type}}
  }

  { // No single-argument constructor.
    using C = Container</*HasDefaultCtr=*/true, /*HasSingleArgCtr=*/false>;
    (void)std::ranges::to<C>(in, 1); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}ranges::to: unable to convert to the given container type}}
  }

  { // No `insert` and no `push_back`.
    using C = Container</*HasDefaultCtr=*/true, /*HasSingleArgCtr=*/true,
                        /*HasInsert=*/false, /*HasInsertWithRightSignature=*/false,
                        /*HasPushBack=*/false, /*HasPushBackWithRightSignature=*/false>;
    (void)std::ranges::to<C>(in); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}ranges::to: unable to convert to the given container type}}
  }

  { // No `push_back`, `insert` has a wrong signature.
    using C = Container</*HasDefaultCtr=*/true, /*HasSingleArgCtr=*/true,
                        /*HasInsert=*/true, /*HasInsertWithRightSignature=*/false,
                        /*HasPushBack=*/false, /*HasPushBackWithRightSignature=*/false>;
    (void)std::ranges::to<C>(in); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}ranges::to: unable to convert to the given container type}}
  }

  { // No `insert`, `push_back` has a wrong signature.
    using C = Container</*HasDefaultCtr=*/true, /*HasSingleArgCtr=*/true,
                        /*HasInsert=*/false, /*HasInsertWithRightSignature=*/false,
                        /*HasPushBack=*/true, /*HasPushBackWithRightSignature=*/false>;
    (void)std::ranges::to<C>(in); //expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}ranges::to: unable to convert to the given container type}}
  }
}
