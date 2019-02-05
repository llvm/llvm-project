//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <iterator>

// back_insert_iterator

// requires CopyConstructible<Cont::value_type>
//   back_insert_iterator<Cont>&
//   operator=(Cont::value_type&& value);

#include <iterator>

#include <vector>
#include <memory>
#include <cassert>

template <class C>
void
test(C c)
{
    std::back_insert_iterator<C> i(c);
    i = typename C::value_type();
    assert(c.back() == typename C::value_type());
}

int main(int, char**)
{
    test(std::vector<std::unique_ptr<int> >());

  return 0;
}
