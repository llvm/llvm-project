//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include <memory>

// void* align(size_t alignment, size_t size, void*& ptr, size_t& space);

#include <memory>
#include <cassert>

int main(int, char**)
{
    const unsigned N = 20;
    char buf[N];
    void* r;
    void* p = &buf[0];
    std::size_t s = N;
    r = std::align(4, 10, p, s);
    assert(p == &buf[0]);
    assert(r == p);
    assert(s == N);

    p = &buf[1];
    s = N;
    r = std::align(4, 10, p, s);
    assert(p == &buf[4]);
    assert(r == p);
    assert(s == N-3);

    p = &buf[2];
    s = N;
    r = std::align(4, 10, p, s);
    assert(p == &buf[4]);
    assert(r == p);
    assert(s == N-2);

    p = &buf[3];
    s = N;
    r = std::align(4, 10, p, s);
    assert(p == &buf[4]);
    assert(r == p);
    assert(s == N-1);

    p = &buf[4];
    s = N;
    r = std::align(4, 10, p, s);
    assert(p == &buf[4]);
    assert(r == p);
    assert(s == N);

    p = &buf[0];
    s = N;
    r = std::align(4, N, p, s);
    assert(p == &buf[0]);
    assert(r == p);
    assert(s == N);

    p = &buf[1];
    s = N-1;
    r = std::align(4, N-4, p, s);
    assert(p == &buf[4]);
    assert(r == p);
    assert(s == N-4);

    p = &buf[1];
    s = N-1;
    r = std::align(4, N-3, p, s);
    assert(p == &buf[1]);
    assert(r == nullptr);
    assert(s == N-1);

    p = &buf[0];
    s = N;
    r = std::align(1, N+1, p, s);
    assert(p == &buf[0]);
    assert(r == nullptr);
    assert(s == N);

  return 0;
}
