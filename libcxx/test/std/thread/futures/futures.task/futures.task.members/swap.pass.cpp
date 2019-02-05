//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <future>

// class packaged_task<R(ArgTypes...)>

// void swap(packaged_task& other);

#include <future>
#include <cassert>

class A
{
    long data_;

public:
    explicit A(long i) : data_(i) {}

    long operator()(long i, long j) const {return data_ + i + j;}
};

int main(int, char**)
{
    {
        std::packaged_task<double(int, char)> p0(A(5));
        std::packaged_task<double(int, char)> p;
        p.swap(p0);
        assert(!p0.valid());
        assert(p.valid());
        std::future<double> f = p.get_future();
        p(3, 'a');
        assert(f.get() == 105.0);
    }
    {
        std::packaged_task<double(int, char)> p0;
        std::packaged_task<double(int, char)> p;
        p.swap(p0);
        assert(!p0.valid());
        assert(!p.valid());
    }

  return 0;
}
