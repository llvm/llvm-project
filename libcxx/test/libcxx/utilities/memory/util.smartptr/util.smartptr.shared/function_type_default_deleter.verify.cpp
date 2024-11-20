//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <memory>
#include <cstddef>
#include <type_traits>

template <int> struct Tag {};

template <int ID>
using SPtr = std::shared_ptr<void(Tag<ID>)>;

template <int ID>
using FnType = void(Tag<ID>);

template <int ID>
void TestFn(Tag<ID>) {}

template <int ID>
FnType<ID>* getFn() {
  return &TestFn<ID>;
}

struct Deleter {
  template <class Tp>
  void operator()(Tp) const {
    using RawT = typename std::remove_pointer<Tp>::type;
    static_assert(std::is_function<RawT>::value ||
                  std::is_same<typename std::remove_cv<RawT>::type,
                               std::nullptr_t>::value,
                  "");
  }
};

int main(int, char**) {
  {
    SPtr<0> s; // OK
    SPtr<1> s1(nullptr); // OK
    SPtr<2> s2(getFn<2>(), Deleter{}); // OK
    SPtr<3> s3(nullptr, Deleter{}); // OK
  }

  // expected-error-re@*:* {{static assertion failed{{.*}}default_delete cannot be instantiated for function types}}
  std::default_delete<FnType<5>> deleter{}; // expected-note {{requested here}}

  return 0;
}
