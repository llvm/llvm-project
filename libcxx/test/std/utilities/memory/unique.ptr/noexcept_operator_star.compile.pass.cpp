//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// unique_ptr

// add_lvalue_reference_t<T> operator*() const noexcept(noexcept(*declval<pointer>()));

// Dereferencing pointer directly in noexcept fails for a void pointer.  This
// is not SFINAE-ed away leading to a hard error. The issue was originally
// triggered by
// test/std/utilities/memory/unique.ptr/iterator_concept_conformance.compile.pass.cpp
//
// This test validates whether the code compiles.

#include <memory>

extern const std::unique_ptr<void> p;
void f() { [[maybe_unused]] bool b = noexcept(p.operator*()); }
