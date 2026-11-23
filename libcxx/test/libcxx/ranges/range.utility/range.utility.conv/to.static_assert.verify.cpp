//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated-volatile

// Test that the "mandates" requirements on the given container are checked using `static_assert`.

#include <ranges>
#include <vector>

void test_cv_qualifications() {
  using R = std::vector<int>;
  R in    = {1, 2, 3};

  //expected-error-re@*:* {{static assertion failed{{.*}}The target container cannot be const-qualified, please remove the const}}
  (void)std::ranges::to<const R>(in);
  //expected-error-re@*:* {{static assertion failed{{.*}}The target container cannot be const-qualified, please remove the const}}
  (void)(in | std::ranges::to<const R>());

  //expected-error-re@*:* {{static assertion failed{{.*}}The target container cannot be volatile-qualified, please remove the volatile}}
  (void)std::ranges::to<volatile R>(in);

  //expected-error-re@*:* {{static assertion failed{{.*}}The target container cannot be volatile-qualified, please remove the volatile}}
  (void)(in | std::ranges::to<volatile R>());
}
//unexpected_types
void ff();
void test_unexpected_types() {
  struct C {
    int member;
    int f();
  };

  enum color { red, green, blue };
  using member_func_ptr = decltype(&C::f);
  using member_ptr      = decltype(&C::member);
  using func_ptr        = decltype(&ff);
  using func_t          = decltype(ff);

  struct R {
    int* begin() const { return nullptr; };
    int* end() const { return nullptr; };

    operator int() const;
    operator int*() const;
    operator func_ptr() const;
    operator member_func_ptr() const;
    operator member_ptr() const;
    operator color() const;
  };
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<int>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<int>());

  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<int*>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<int*>());

  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<func_ptr>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<func_ptr>());

  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<member_ptr>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<member_ptr>());

  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<func_t>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<func_t>());

  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<void>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  //expected-error-re@*:* {{static assertion failed{{.*}}ranges::to: unable to convert to the given container type.}}
  (void)(R{} | std::ranges::to<void>());

  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<color>(R{});
  //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<color>());
}
