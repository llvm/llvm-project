//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14
// UNSUPPORTED: no-threads

// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

// test multithreaded container debugging

#include <cassert>
#include <cstddef>
#include <deque>
#include <list>
#include <thread>
#include <vector>

template <typename Container>
Container makeContainer(int size) {
  Container c;
  typedef typename Container::value_type ValueType;
  for (int i = 0; i < size; ++i)
    c.insert(c.end(), ValueType(i));
  assert(c.size() == static_cast<std::size_t>(size));
  return c;
}

template <typename Container>
void ThreadUseIter() {
  const std::size_t maxRounds = 7;
  struct TestRunner{
    void operator()() {
      for (std::size_t count = 0; count < maxRounds; count++) {
        const std::size_t containerCount = 11;
        std::vector<Container> containers;
        std::vector<typename Container::iterator> iterators;
        for (std::size_t containerIndex = 0; containerIndex < containerCount; containerIndex++) {
          containers.push_back(makeContainer<Container>(3));
          Container& c = containers.back();
          iterators.push_back(c.begin());
          iterators.push_back(c.end());
        }
      }
    }
  };

  TestRunner r;
  const std::size_t threadCount = 4;
  std::vector<std::thread> threads;
  for (std::size_t count = 0; count < threadCount; count++)
    threads.emplace_back(r);
  r();
  for (std::size_t count = 0; count < threadCount; count++)
    threads[count].join();
}

int main(int, char**) {
  ThreadUseIter<std::vector<int> >();
  return 0;
}
