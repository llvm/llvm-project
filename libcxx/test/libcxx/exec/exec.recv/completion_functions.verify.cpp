//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-execution

// The noexcept and void-return requirements are mandates, not constraints.

#include <execution>

namespace ex = std::execution;

struct ThrowingReceiver {
  void set_value() && noexcept(false);
  void set_error(int) && noexcept(false);
  void set_stopped() && noexcept(false);
};

struct NonVoidReceiver {
  int set_value() && noexcept;
  int set_error(int) && noexcept;
  int set_stopped() && noexcept;
};

// Objects of this type can be converted to int, but the conversion may throw.
struct ThrowingConversion {
  operator int() noexcept(false);
};

struct ConversionReceiver {
  void set_value(int) && noexcept;
  void set_error(int) && noexcept;
};

void test() {
  // expected-error-re@*:* {{static assertion failed{{.*}}set_value must be noexcept}}
  ex::set_value(ThrowingReceiver{});
  // expected-error-re@*:* {{static assertion failed{{.*}}set_error must be noexcept}}
  ex::set_error(ThrowingReceiver{}, 0);
  // expected-error-re@*:* {{static assertion failed{{.*}}set_stopped must be noexcept}}
  ex::set_stopped(ThrowingReceiver{});

  // expected-error-re@*:* {{static assertion failed{{.*}}set_value must return void}}
  ex::set_value(NonVoidReceiver{});
  // expected-error-re@*:* {{static assertion failed{{.*}}set_error must return void}}
  ex::set_error(NonVoidReceiver{}, 0);
  // expected-error-re@*:* {{static assertion failed{{.*}}set_stopped must return void}}
  ex::set_stopped(NonVoidReceiver{});

  // expected-error-re@*:* {{static assertion failed{{.*}}set_value must be noexcept}}
  ex::set_value(ConversionReceiver{}, ThrowingConversion{});
  // expected-error-re@*:* {{static assertion failed{{.*}}set_error must be noexcept}}
  ex::set_error(ConversionReceiver{}, ThrowingConversion{});
}
