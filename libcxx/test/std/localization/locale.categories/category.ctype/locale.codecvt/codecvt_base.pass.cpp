//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class codecvt_base
// {
// public:
//     enum result {ok, partial, error, noconv};
// };

#include <locale>
#include <cassert>

int main()
{
    assert(std::codecvt_base::ok == 0);
    assert(std::codecvt_base::partial == 1);
    assert(std::codecvt_base::error == 2);
    assert(std::codecvt_base::noconv == 3);
}
