//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// __referenceable<Tp>
//
// [defns.referenceable] defines "a referenceable type" as:
// An object type, a function type that does not have cv-qualifiers
//    or a ref-qualifier, or a reference type.
//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <__concepts/referenceable.h>

struct Foo {};

static_assert(!std::__referenceable<void>);
static_assert(std::__referenceable<int>);
static_assert(std::__referenceable<int[3]>);
static_assert(std::__referenceable<int[]>);
static_assert(std::__referenceable<int&>);
static_assert(std::__referenceable<const int&>);
static_assert(std::__referenceable<int*>);
static_assert(std::__referenceable<const int*>);
static_assert(std::__referenceable<Foo>);
static_assert(std::__referenceable<const Foo>);
static_assert(std::__referenceable<Foo&>);
static_assert(std::__referenceable<const Foo&>);
static_assert(std::__referenceable<Foo&&>);
static_assert(std::__referenceable<const Foo&&>);

static_assert(std::__referenceable<int __attribute__((__vector_size__(8)))>);
static_assert(std::__referenceable<const int __attribute__((__vector_size__(8)))>);
static_assert(std::__referenceable<float __attribute__((__vector_size__(16)))>);
static_assert(std::__referenceable<const float __attribute__((__vector_size__(16)))>);

// Functions without cv-qualifiers are referenceable
static_assert(std::__referenceable<void()>);
static_assert(!std::__referenceable<void() const>);
static_assert(!std::__referenceable<void() &>);
static_assert(!std::__referenceable<void() const&>);
static_assert(!std::__referenceable<void() &&>);
static_assert(!std::__referenceable<void() const&&>);

static_assert(std::__referenceable<void(int)>);
static_assert(!std::__referenceable<void(int) const>);
static_assert(!std::__referenceable<void(int) &>);
static_assert(!std::__referenceable<void(int) const&>);
static_assert(!std::__referenceable<void(int) &&>);
static_assert(!std::__referenceable<void(int) const&&>);

static_assert(std::__referenceable<void(int, float)>);
static_assert(!std::__referenceable<void(int, float) const>);
static_assert(!std::__referenceable<void(int, float) &>);
static_assert(!std::__referenceable<void(int, float) const&>);
static_assert(!std::__referenceable<void(int, float) &&>);
static_assert(!std::__referenceable<void(int, float) const&&>);

static_assert(std::__referenceable<void(int, float, Foo&)>);
static_assert(!std::__referenceable<void(int, float, Foo&) const>);
static_assert(!std::__referenceable<void(int, float, Foo&) &>);
static_assert(!std::__referenceable<void(int, float, Foo&) const&>);
static_assert(!std::__referenceable<void(int, float, Foo&) &&>);
static_assert(!std::__referenceable<void(int, float, Foo&) const&&>);

static_assert(std::__referenceable<void(...)>);
static_assert(!std::__referenceable<void(...) const>);
static_assert(!std::__referenceable<void(...) &>);
static_assert(!std::__referenceable<void(...) const&>);
static_assert(!std::__referenceable<void(...) &&>);
static_assert(!std::__referenceable<void(...) const&&>);

static_assert(std::__referenceable<void(int, ...)>);
static_assert(!std::__referenceable<void(int, ...) const>);
static_assert(!std::__referenceable<void(int, ...) &>);
static_assert(!std::__referenceable<void(int, ...) const&>);
static_assert(!std::__referenceable<void(int, ...) &&>);
static_assert(!std::__referenceable<void(int, ...) const&&>);

static_assert(std::__referenceable<void(int, float, ...)>);
static_assert(!std::__referenceable<void(int, float, ...) const>);
static_assert(!std::__referenceable<void(int, float, ...) &>);
static_assert(!std::__referenceable<void(int, float, ...) const&>);
static_assert(!std::__referenceable<void(int, float, ...) &&>);
static_assert(!std::__referenceable<void(int, float, ...) const&&>);

static_assert(std::__referenceable<void(int, float, Foo&, ...)>);
static_assert(!std::__referenceable<void(int, float, Foo&, ...) const>);
static_assert(!std::__referenceable<void(int, float, Foo&, ...) &>);
static_assert(!std::__referenceable<void(int, float, Foo&, ...) const&>);
static_assert(!std::__referenceable<void(int, float, Foo&, ...) &&>);
static_assert(!std::__referenceable<void(int, float, Foo&, ...) const&&>);

// member functions with or without cv-qualifiers are referenceable
static_assert(std::__referenceable<void (Foo::*)()>);
static_assert(std::__referenceable<void (Foo::*)() const>);
static_assert(std::__referenceable<void (Foo::*)() &>);
static_assert(std::__referenceable<void (Foo::*)() const&>);
static_assert(std::__referenceable<void (Foo::*)() &&>);
static_assert(std::__referenceable<void (Foo::*)() const&&>);

static_assert(std::__referenceable<void (Foo::*)(int)>);
static_assert(std::__referenceable<void (Foo::*)(int) const>);
static_assert(std::__referenceable<void (Foo::*)(int) &>);
static_assert(std::__referenceable<void (Foo::*)(int) const&>);
static_assert(std::__referenceable<void (Foo::*)(int) &&>);
static_assert(std::__referenceable<void (Foo::*)(int) const&&>);

static_assert(std::__referenceable<void (Foo::*)(int, float)>);
static_assert(std::__referenceable<void (Foo::*)(int, float) const>);
static_assert(std::__referenceable<void (Foo::*)(int, float) &>);
static_assert(std::__referenceable<void (Foo::*)(int, float) const&>);
static_assert(std::__referenceable<void (Foo::*)(int, float) &&>);
static_assert(std::__referenceable<void (Foo::*)(int, float) const&&>);

static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&)>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&) const>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&) &>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&) const&>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&) &&>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&) const&&>);

static_assert(std::__referenceable<void (Foo::*)(...)>);
static_assert(std::__referenceable<void (Foo::*)(...) const>);
static_assert(std::__referenceable<void (Foo::*)(...) &>);
static_assert(std::__referenceable<void (Foo::*)(...) const&>);
static_assert(std::__referenceable<void (Foo::*)(...) &&>);
static_assert(std::__referenceable<void (Foo::*)(...) const&&>);

static_assert(std::__referenceable<void (Foo::*)(int, ...)>);
static_assert(std::__referenceable<void (Foo::*)(int, ...) const>);
static_assert(std::__referenceable<void (Foo::*)(int, ...) &>);
static_assert(std::__referenceable<void (Foo::*)(int, ...) const&>);
static_assert(std::__referenceable<void (Foo::*)(int, ...) &&>);
static_assert(std::__referenceable<void (Foo::*)(int, ...) const&&>);

static_assert(std::__referenceable<void (Foo::*)(int, float, ...)>);
static_assert(std::__referenceable<void (Foo::*)(int, float, ...) const>);
static_assert(std::__referenceable<void (Foo::*)(int, float, ...) &>);
static_assert(std::__referenceable<void (Foo::*)(int, float, ...) const&>);
static_assert(std::__referenceable<void (Foo::*)(int, float, ...) &&>);
static_assert(std::__referenceable<void (Foo::*)(int, float, ...) const&&>);

static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&, ...)>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&, ...) const>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&, ...) &>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&, ...) const&>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&, ...) &&>);
static_assert(std::__referenceable<void (Foo::*)(int, float, Foo&, ...) const&&>);
