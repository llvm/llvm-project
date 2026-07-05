//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

template <typename T>
struct with_public_dtor : T
{
    template <typename... Args>
    explicit with_public_dtor(Args &&... args)
        : T(std::forward<Args>(args)...)
    {
    }
};
