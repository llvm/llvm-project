//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T, class D>
// struct hash<unique_ptr<T, D>>
// {
//     typedef unique_ptr<T, D> argument_type;
//     typedef size_t           result_type;
//     size_t operator()(const unique_ptr<T, D>& p) const;
// };

#include <memory>
#include <cassert>

int main()
{
    int* ptr = new int;
    std::unique_ptr<int> p(ptr);
    std::hash<std::unique_ptr<int> > f;
    std::size_t h = f(p);
    assert(h == std::hash<int*>()(ptr));
}
