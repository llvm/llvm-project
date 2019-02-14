//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <thread>

// class thread::id

// bool operator< (thread::id x, thread::id y);
// bool operator<=(thread::id x, thread::id y);
// bool operator> (thread::id x, thread::id y);
// bool operator>=(thread::id x, thread::id y);

#include <thread>
#include <cassert>

int main(int, char**)
{
    std::thread::id id0;
    std::thread::id id1;
    std::thread::id id2 = std::this_thread::get_id();
    assert(!(id0 <  id1));
    assert( (id0 <= id1));
    assert(!(id0 >  id1));
    assert( (id0 >= id1));
    assert(!(id0 == id2));
    if (id0 < id2) {
      assert( (id0 <= id2));
      assert(!(id0 >  id2));
      assert(!(id0 >= id2));
    } else {
      assert(!(id0 <= id2));
      assert( (id0 >  id2));
      assert( (id0 >= id2));
    }

  return 0;
}
