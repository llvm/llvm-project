//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <type_traits>

// __is_implicitly_default_constructible<Tp>

#include <type_traits>


struct ExplicitlyDefaultConstructible1 {
    explicit ExplicitlyDefaultConstructible1() = default;
};

struct ExplicitlyDefaultConstructible2 {
    explicit ExplicitlyDefaultConstructible2() { }
};

struct ImplicitlyDefaultConstructible1 {
    ImplicitlyDefaultConstructible1() { }
};

struct ImplicitlyDefaultConstructible2 {
    ImplicitlyDefaultConstructible2() = default;
};

struct NonDefaultConstructible1 {
    NonDefaultConstructible1() = delete;
};

struct NonDefaultConstructible2 {
    explicit NonDefaultConstructible2() = delete;
};

struct NonDefaultConstructible3 {
    NonDefaultConstructible3(NonDefaultConstructible3&&) { }
};

struct ProtectedDefaultConstructible {
protected:
    ProtectedDefaultConstructible() = default;
};

struct PrivateDefaultConstructible {
private:
    PrivateDefaultConstructible() = default;
};

struct Base { };

struct ProtectedDefaultConstructibleWithBase : Base {
protected:
    ProtectedDefaultConstructibleWithBase() = default;
};

struct PrivateDefaultConstructibleWithBase : Base {
private:
    PrivateDefaultConstructibleWithBase() = default;
};

static_assert(!std::__is_implicitly_default_constructible<ExplicitlyDefaultConstructible1>::value, "");
static_assert(!std::__is_implicitly_default_constructible<ExplicitlyDefaultConstructible2>::value, "");
static_assert(std::__is_implicitly_default_constructible<ImplicitlyDefaultConstructible1>::value, "");
static_assert(std::__is_implicitly_default_constructible<ImplicitlyDefaultConstructible2>::value, "");
static_assert(!std::__is_implicitly_default_constructible<NonDefaultConstructible1>::value, "");
static_assert(!std::__is_implicitly_default_constructible<NonDefaultConstructible2>::value, "");
static_assert(!std::__is_implicitly_default_constructible<NonDefaultConstructible3>::value, "");
static_assert(!std::__is_implicitly_default_constructible<ProtectedDefaultConstructible>::value, "");
static_assert(!std::__is_implicitly_default_constructible<PrivateDefaultConstructible>::value, "");
static_assert(!std::__is_implicitly_default_constructible<ProtectedDefaultConstructibleWithBase>::value, "");
static_assert(!std::__is_implicitly_default_constructible<PrivateDefaultConstructibleWithBase>::value, "");

int main(int, char**) {
    return 0;
}
