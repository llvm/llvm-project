//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept move_constructible;

#include <concepts>
#include <type_traits>

#include "type_classification/moveconstructible.h"

static_assert(std::move_constructible<int>);
static_assert(std::move_constructible<int*>);
static_assert(std::move_constructible<int&>);
static_assert(std::move_constructible<int&&>);
static_assert(std::move_constructible<const int>);
static_assert(std::move_constructible<const int&>);
static_assert(std::move_constructible<const int&&>);
static_assert(std::move_constructible<volatile int>);
static_assert(std::move_constructible<volatile int&>);
static_assert(std::move_constructible<volatile int&&>);
static_assert(std::move_constructible<int (*)()>);
static_assert(std::move_constructible<int (&)()>);
static_assert(std::move_constructible<HasDefaultOps>);
static_assert(std::move_constructible<CustomMoveCtor>);
static_assert(std::move_constructible<MoveOnly>);
static_assert(std::move_constructible<const CustomMoveCtor&>);
static_assert(std::move_constructible<volatile CustomMoveCtor&>);
static_assert(std::move_constructible<const CustomMoveCtor&&>);
static_assert(std::move_constructible<volatile CustomMoveCtor&&>);
static_assert(std::move_constructible<CustomMoveAssign>);
static_assert(std::move_constructible<const CustomMoveAssign&>);
static_assert(std::move_constructible<volatile CustomMoveAssign&>);
static_assert(std::move_constructible<const CustomMoveAssign&&>);
static_assert(std::move_constructible<volatile CustomMoveAssign&&>);
static_assert(std::move_constructible<int HasDefaultOps::*>);
static_assert(std::move_constructible<void (HasDefaultOps::*)(int)>);
static_assert(std::move_constructible<MemberLvalueReference>);
static_assert(std::move_constructible<MemberRvalueReference>);

static_assert(!std::move_constructible<void>);
static_assert(!std::move_constructible<const CustomMoveCtor>);
static_assert(!std::move_constructible<volatile CustomMoveCtor>);
static_assert(!std::move_constructible<const CustomMoveAssign>);
static_assert(!std::move_constructible<volatile CustomMoveAssign>);
static_assert(!std::move_constructible<int[10]>);
static_assert(!std::move_constructible<DeletedMoveCtor>);
static_assert(!std::move_constructible<ImplicitlyDeletedMoveCtor>);
static_assert(!std::move_constructible<DeletedMoveAssign>);
static_assert(!std::move_constructible<ImplicitlyDeletedMoveAssign>);

static_assert(std::move_constructible<DeletedMoveCtor&>);
static_assert(std::move_constructible<DeletedMoveCtor&&>);
static_assert(std::move_constructible<const DeletedMoveCtor&>);
static_assert(std::move_constructible<const DeletedMoveCtor&&>);
static_assert(std::move_constructible<ImplicitlyDeletedMoveCtor&>);
static_assert(std::move_constructible<ImplicitlyDeletedMoveCtor&&>);
static_assert(std::move_constructible<const ImplicitlyDeletedMoveCtor&>);
static_assert(std::move_constructible<const ImplicitlyDeletedMoveCtor&&>);
static_assert(std::move_constructible<DeletedMoveAssign&>);
static_assert(std::move_constructible<DeletedMoveAssign&&>);
static_assert(std::move_constructible<const DeletedMoveAssign&>);
static_assert(std::move_constructible<const DeletedMoveAssign&&>);
static_assert(std::move_constructible<ImplicitlyDeletedMoveAssign&>);
static_assert(std::move_constructible<ImplicitlyDeletedMoveAssign&&>);
static_assert(std::move_constructible<const ImplicitlyDeletedMoveAssign&>);
static_assert(std::move_constructible<const ImplicitlyDeletedMoveAssign&&>);

static_assert(!std::move_constructible<NonMovable>);
static_assert(!std::move_constructible<DerivedFromNonMovable>);
static_assert(!std::move_constructible<HasANonMovable>);

int main(int, char**) { return 0; }
