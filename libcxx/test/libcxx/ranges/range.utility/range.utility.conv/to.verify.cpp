//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that the "mandates" requirements on the given class are checked using `static_assert`.
#include <ranges>
void ff() {}
void test() {
  struct C {
    int member;
    int f() { return 0; }
  };

  enum color { red, green, blue };
  using member_func_ptr = decltype(&C::f);
  using member_ptr      = decltype(&C::member);
  using func_ptr        = decltype(&ff);

  struct R {
    int* begin() const { return nullptr; };
    int* end() const { return nullptr; };

    operator int() const { return 0; }
    operator int*() const { return nullptr; }
    operator func_ptr() const { return nullptr; }
    operator member_func_ptr() const { return nullptr; }
    operator member_ptr() const { return nullptr; }
    operator color() const { return color::red; }
  };
  (void)std::ranges::to<int>(
      R{}); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<
                   int>()); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<int*>(
      R{}); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<
                   int*>()); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)std::ranges::to<func_ptr>(
      R{}); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} |
         std::ranges::to<
             func_ptr>()); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}

  (void)std::ranges::to<member_ptr>(
      R{}); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} |
         std::ranges::to<
             member_ptr>()); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}

  (void)std::ranges::to<func_t>(
      R{}); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<
                   func_t>()); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}

  (void)std::ranges::to<void>(
      R{}); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<
                   void>()); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  //expected-error-re@*:* {{static assertion failed{{.*}}ranges::to: unable to convert to the given container type.}}

  (void)std::ranges::to<color>(
      R{}); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
  (void)(R{} | std::ranges::to<
                   color>()); //expected-error-re@*:* {{static assertion failed{{.*}}The target must be a class type}}
}