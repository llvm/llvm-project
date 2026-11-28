//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// This test pins down the current libc++ behavior that vector2<T>::empty() can be
// called even when T is an incomplete type. The standard does not require this:
// [vector2.overview] only guarantees that an incomplete type may be used to
// instantiate vector2, and requires the type to be complete before any method is
// called.
//
// However, libc++ made that work previously, and this test pins down that behavior
// to avoid breaking it unintentionally. Note that this is not a guarantee to users
// that we will support this in the future: this merely guards against changing this
// behavior unknowingly.

#include <vector>

struct Incomplete;

bool call_empty(std::vector2<Incomplete>& v) { return v.empty(); }
bool call_empty_const(const std::vector2<Incomplete>& v) { return v.empty(); }
