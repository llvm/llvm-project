//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// PROTOTYPE: P3866R0 (V2: An Evolution Path for the Standard Library)
//
// Surprising consequences of implementing release namespaces with a using-directive.

#include <type_traits>
#include <vector>

// Every name in std is reachable through every release namespace, including the versioned names themselves and
// the other release namespaces.
static_assert(std::is_same_v<std::cpp29::vector2<int>, std::vector2<int>>);
static_assert(std::is_same_v<std::cpp26::vector2<int>, std::vector2<int>>); // C++26 never had vector2!
static_assert(std::is_same_v<std::cpp29::cpp26::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<std::cpp26::cpp29::vector<int>, std::vector2<int>>);

// std::cpp29::vector is an alias template, not a class template. When used as a template template argument, it
// is not equivalent to std::vector2 (see CWG1286).
template <template <class...> class>
struct wrap {};
static_assert(!std::is_same_v<wrap<std::cpp29::vector>, wrap<std::vector2>>);

// Similarly, something like is_specialization_of<T, std::cpp29::vector> wouldn't work, since a specialization
// of an alias template is just its target type.
template <class T, template <class...> class Tmpl>
inline constexpr bool is_specialization_of = false;
template <template <class...> class Tmpl, class... Args>
inline constexpr bool is_specialization_of<Tmpl<Args...>, Tmpl> = true;
static_assert(is_specialization_of<std::cpp29::vector<int>, std::vector2>);
static_assert(!is_specialization_of<std::cpp29::vector<int>, std::cpp29::vector>);
