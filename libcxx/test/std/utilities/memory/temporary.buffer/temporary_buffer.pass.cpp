//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T>
//   pair<T*, ptrdiff_t>
//   get_temporary_buffer(ptrdiff_t n);
//
// template <class T>
//   void
//   return_temporary_buffer(T* p);

#include <memory>
#include <cassert>

int main()
{
    std::pair<int*, std::ptrdiff_t> ip = std::get_temporary_buffer<int>(5);
    assert(ip.first);
    assert(ip.second == 5);
    std::return_temporary_buffer(ip.first);
}
