// RUN: %clangxx %s -o %t
// RUN: %t | FileCheck %s

#include <iostream>

class A {};
class B : public A {};

template <class _Base, class _Derived>
inline constexpr bool is_pointer_interconvertible_base_of_v = __is_pointer_interconvertible_base_of(_Base, _Derived);

int main() {
    // CHECK: 1
    std::cout << __is_pointer_interconvertible_base_of(const A, A) << std::endl;
    // CHECK-NEXT: 1
    std::cout << is_pointer_interconvertible_base_of_v<const A, A> << std::endl;

    // CHECK-NEXT: 1
    std::cout << __is_pointer_interconvertible_base_of(const A, B) << std::endl;
    // CHECK-NEXT: 1
    std::cout << is_pointer_interconvertible_base_of_v<const A, B> << std::endl;
}
