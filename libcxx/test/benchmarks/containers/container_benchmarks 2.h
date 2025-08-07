// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_BENCHMARKS_CONTAINERS_CONTAINER_BENCHMARKS_H
#define TEST_BENCHMARKS_CONTAINERS_CONTAINER_BENCHMARKS_H

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <ranges> // for std::from_range
#include <string>
#include <type_traits>
#include <vector>

#include "benchmark/benchmark.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "../GenerateInput.h"

namespace ContainerBenchmarks {

template <class Container>
void DoNotOptimizeData(Container& c) {
  if constexpr (requires { c.data(); }) {
    benchmark::DoNotOptimize(c.data());
  } else {
    benchmark::DoNotOptimize(&c);
  }
}

//
// Sequence container operations
//
template <class Container>
void BM_ctor_size(benchmark::State& st) {
  auto size = st.range(0);

  for (auto _ : st) {
    Container c(size); // we assume the destructor doesn't dominate the benchmark
    DoNotOptimizeData(c);
  }
}

template <class Container, class Generator>
void BM_ctor_size_value(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const auto size = st.range(0);
  ValueType value = gen();
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    Container c(size, value); // we assume the destructor doesn't dominate the benchmark
    DoNotOptimizeData(c);
  }
}

template <class Container, class Generator>
void BM_ctor_iter_iter(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const auto size = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  const auto begin = in.begin();
  const auto end   = in.end();
  benchmark::DoNotOptimize(in);

  for (auto _ : st) {
    Container c(begin, end); // we assume the destructor doesn't dominate the benchmark
    DoNotOptimizeData(c);
  }
}

#if TEST_STD_VER >= 23
template <class Container, class Generator>
void BM_ctor_from_range(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const auto size = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  benchmark::DoNotOptimize(in);

  for (auto _ : st) {
    Container c(std::from_range, in); // we assume the destructor doesn't dominate the benchmark
    DoNotOptimizeData(c);
  }
}
#endif

template <class Container, class Generator>
void BM_ctor_copy(benchmark::State& st, Generator gen) {
  auto size = st.range(0);
  Container in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);

  for (auto _ : st) {
    Container c(in); // we assume the destructor doesn't dominate the benchmark
    DoNotOptimizeData(c);
    DoNotOptimizeData(in);
  }
}

template <class Container, class Generator>
void BM_assignment(benchmark::State& st, Generator gen) {
  auto size = st.range(0);
  Container in1, in2;
  std::generate_n(std::back_inserter(in1), size, gen);
  std::generate_n(std::back_inserter(in2), size, gen);
  DoNotOptimizeData(in1);
  DoNotOptimizeData(in2);

  // Assign from one of two containers in succession to avoid
  // hitting a self-assignment corner-case
  Container c(in1);
  bool toggle = false;
  for (auto _ : st) {
    c      = toggle ? in1 : in2;
    toggle = !toggle;
    DoNotOptimizeData(c);
    DoNotOptimizeData(in1);
    DoNotOptimizeData(in2);
  }
}

// Benchmark Container::assign(input-iter, input-iter) when the container already contains
// the same number of elements that we're assigning. The intent is to check whether the
// implementation basically creates a new container from scratch or manages to reuse the
// pre-existing storage.
template <typename Container, class Generator>
void BM_assign_input_iter_full(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  auto size       = st.range(0);
  std::vector<ValueType> in1, in2;
  std::generate_n(std::back_inserter(in1), size, gen);
  std::generate_n(std::back_inserter(in2), size, gen);
  DoNotOptimizeData(in1);
  DoNotOptimizeData(in2);

  Container c(in1.begin(), in1.end());
  bool toggle = false;
  for (auto _ : st) {
    std::vector<ValueType>& in = toggle ? in1 : in2;
    auto first                 = in.data();
    auto last                  = in.data() + in.size();
    c.assign(cpp17_input_iterator(first), cpp17_input_iterator(last));
    toggle = !toggle;
    DoNotOptimizeData(c);
  }
}

template <class Container, class Generator>
void BM_insert_begin(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);

  Container c(in.begin(), in.end());
  DoNotOptimizeData(c);

  ValueType value = gen();
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    c.insert(c.begin(), value);
    DoNotOptimizeData(c);

    c.erase(std::prev(c.end())); // avoid growing indefinitely
  }
}

template <class Container, class Generator>
  requires std::random_access_iterator<typename Container::iterator>
void BM_insert_middle(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);

  Container c(in.begin(), in.end());
  DoNotOptimizeData(c);

  ValueType value = gen();
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    auto mid = c.begin() + (size / 2); // requires random-access iterators in order to make sense
    c.insert(mid, value);
    DoNotOptimizeData(c);

    c.erase(c.end() - 1); // avoid growing indefinitely
  }
}

// Insert at the start of a vector in a scenario where the vector already
// has enough capacity to hold all the elements we are inserting.
template <class Container, class Generator>
void BM_insert_begin_input_iter_with_reserve_no_realloc(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);
  auto first = in.data();
  auto last  = in.data() + in.size();

  const int small = 100; // arbitrary
  Container c;
  c.reserve(size + small); // ensure no reallocation
  std::generate_n(std::back_inserter(c), small, gen);

  for (auto _ : st) {
    c.insert(c.begin(), cpp17_input_iterator(first), cpp17_input_iterator(last));
    DoNotOptimizeData(c);

    st.PauseTiming();
    c.erase(c.begin() + small, c.end()); // avoid growing indefinitely
    st.ResumeTiming();
  }
}

// Insert at the start of a vector in a scenario where the vector already
// has almost enough capacity to hold all the elements we are inserting,
// but does need to reallocate.
template <class Container, class Generator>
void BM_insert_begin_input_iter_with_reserve_almost_no_realloc(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);
  auto first = in.data();
  auto last  = in.data() + in.size();

  const int overflow = size / 10; // 10% of elements won't fit in the vector when we insert
  Container c;
  for (auto _ : st) {
    st.PauseTiming();
    c = Container();
    c.reserve(size);
    std::generate_n(std::back_inserter(c), overflow, gen);
    st.ResumeTiming();

    c.insert(c.begin(), cpp17_input_iterator(first), cpp17_input_iterator(last));
    DoNotOptimizeData(c);
  }
}

// Insert at the start of a vector in a scenario where the vector can fit a few
// more elements, but needs to reallocate almost immediately to fit the remaining
// elements.
template <class Container, class Generator>
void BM_insert_begin_input_iter_with_reserve_near_full(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);
  auto first = in.data();
  auto last  = in.data() + in.size();

  const int overflow = 9 * (size / 10); // 90% of elements won't fit in the vector when we insert
  Container c;
  for (auto _ : st) {
    st.PauseTiming();
    c = Container();
    c.reserve(size);
    std::generate_n(std::back_inserter(c), overflow, gen);
    st.ResumeTiming();

    c.insert(c.begin(), cpp17_input_iterator(first), cpp17_input_iterator(last));
    DoNotOptimizeData(c);
  }
}

template <class Container, class Generator>
void BM_erase_begin(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);

  Container c(in.begin(), in.end());
  DoNotOptimizeData(c);

  ValueType value = gen();
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    c.erase(c.begin());
    DoNotOptimizeData(c);

    c.insert(c.end(), value); // re-insert an element at the end to avoid needing a new container
  }
}

template <class Container, class Generator>
  requires std::random_access_iterator<typename Container::iterator>
void BM_erase_middle(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);

  Container c(in.begin(), in.end());
  DoNotOptimizeData(c);

  ValueType value = gen();
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    auto mid = c.begin() + (size / 2);
    c.erase(mid);
    DoNotOptimizeData(c);

    c.insert(c.end(), value); // re-insert an element at the end to avoid needing a new container
  }
}

template <class Container, class Generator>
void BM_push_back(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);

  Container c;
  DoNotOptimizeData(c);
  while (st.KeepRunningBatch(size)) {
    c.clear();
    for (int i = 0; i != size; ++i) {
      c.push_back(in[i]);
    }
    DoNotOptimizeData(c);
  }
}

template <class Container, class Generator>
void BM_push_back_with_reserve(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const int size  = st.range(0);
  std::vector<ValueType> in;
  std::generate_n(std::back_inserter(in), size, gen);
  DoNotOptimizeData(in);

  Container c;
  c.reserve(size);
  DoNotOptimizeData(c);
  while (st.KeepRunningBatch(size)) {
    c.clear();
    for (int i = 0; i != size; ++i) {
      c.push_back(in[i]);
    }
    DoNotOptimizeData(c);
  }
}

template <class Container>
void sequence_container_benchmarks(std::string container) {
  using ValueType = typename Container::value_type;

  using Generator     = ValueType (*)();
  Generator cheap     = [] { return Generate<ValueType>::cheap(); };
  Generator expensive = [] { return Generate<ValueType>::expensive(); };
  auto tostr          = [&](Generator gen) { return gen == cheap ? " (cheap elements)" : " (expensive elements)"; };
  std::vector<Generator> generators;
  generators.push_back(cheap);
  if constexpr (!std::is_integral_v<ValueType>) {
    generators.push_back(expensive);
  }

  // constructors
  if constexpr (std::is_constructible_v<Container, std::size_t>) {
    // not all containers provide this one
    benchmark::RegisterBenchmark(container + "::ctor(size)", BM_ctor_size<Container>)->Arg(1024);
  }
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::ctor(size, value_type)" + tostr(gen), [=](auto& st) {
      BM_ctor_size_value<Container>(st, gen);
    })->Arg(1024);
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::ctor(Iterator, Iterator)" + tostr(gen), [=](auto& st) {
      BM_ctor_iter_iter<Container>(st, gen);
    })->Arg(1024);
#if TEST_STD_VER >= 23
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::ctor(Range)" + tostr(gen), [=](auto& st) {
      BM_ctor_from_range<Container>(st, gen);
    })->Arg(1024);
#endif
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::ctor(const&)" + tostr(gen), [=](auto& st) {
      BM_ctor_copy<Container>(st, gen);
    })->Arg(1024);

  // assignment
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::operator=(const&)" + tostr(gen), [=](auto& st) {
      BM_assignment<Container>(st, gen);
    })->Arg(1024);
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::assign(input-iter, input-iter) (full container)" + tostr(gen),
                                 [=](auto& st) { BM_assign_input_iter_full<Container>(st, gen); })
        ->Arg(1024);

  // insert
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::insert(begin)" + tostr(gen), [=](auto& st) {
      BM_insert_begin<Container>(st, gen);
    })->Arg(1024);
  if constexpr (std::random_access_iterator<typename Container::iterator>) {
    for (auto gen : generators)
      benchmark::RegisterBenchmark(container + "::insert(middle)" + tostr(gen), [=](auto& st) {
        BM_insert_middle<Container>(st, gen);
      })->Arg(1024);
  }
  if constexpr (requires(Container c) { c.reserve(0); }) {
    for (auto gen : generators)
      benchmark::RegisterBenchmark(
          container + "::insert(begin, input-iter, input-iter) (no realloc)" + tostr(gen),
          [=](auto& st) { BM_insert_begin_input_iter_with_reserve_no_realloc<Container>(st, gen); })
          ->Arg(1024);
    for (auto gen : generators)
      benchmark::RegisterBenchmark(
          container + "::insert(begin, input-iter, input-iter) (half filled)" + tostr(gen),
          [=](auto& st) { BM_insert_begin_input_iter_with_reserve_almost_no_realloc<Container>(st, gen); })
          ->Arg(1024);
    for (auto gen : generators)
      benchmark::RegisterBenchmark(
          container + "::insert(begin, input-iter, input-iter) (near full)" + tostr(gen),
          [=](auto& st) { BM_insert_begin_input_iter_with_reserve_near_full<Container>(st, gen); })
          ->Arg(1024);
  }

  // erase
  for (auto gen : generators)
    benchmark::RegisterBenchmark(container + "::erase(begin)" + tostr(gen), [=](auto& st) {
      BM_erase_begin<Container>(st, gen);
    })->Arg(1024);
  if constexpr (std::random_access_iterator<typename Container::iterator>) {
    for (auto gen : generators)
      benchmark::RegisterBenchmark(container + "::erase(middle)" + tostr(gen), [=](auto& st) {
        BM_erase_middle<Container>(st, gen);
      })->Arg(1024);
  }

  // push_back (optional)
  if constexpr (requires(Container c, ValueType v) { c.push_back(v); }) {
    for (auto gen : generators)
      benchmark::RegisterBenchmark(container + "::push_back()" + tostr(gen), [=](auto& st) {
        BM_push_back<Container>(st, gen);
      })->Arg(1024);
    if constexpr (requires(Container c) { c.reserve(0); }) {
      for (auto gen : generators)
        benchmark::RegisterBenchmark(container + "::push_back() (with reserve)" + tostr(gen), [=](auto& st) {
          BM_push_back_with_reserve<Container>(st, gen);
        })->Arg(1024);
    }
  }
}

//
// Misc operations
//
template <class Container, class GenInputs>
void BM_InsertValue(benchmark::State& st, Container c, GenInputs gen) {
  auto in        = gen(st.range(0));
  const auto end = in.end();
  while (st.KeepRunning()) {
    c.clear();
    for (auto it = in.begin(); it != end; ++it) {
      benchmark::DoNotOptimize(&(*c.insert(*it).first));
    }
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_InsertValueRehash(benchmark::State& st, Container c, GenInputs gen) {
  auto in        = gen(st.range(0));
  const auto end = in.end();
  while (st.KeepRunning()) {
    c.clear();
    c.rehash(16);
    for (auto it = in.begin(); it != end; ++it) {
      benchmark::DoNotOptimize(&(*c.insert(*it).first));
    }
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_InsertDuplicate(benchmark::State& st, Container c, GenInputs gen) {
  auto in        = gen(st.range(0));
  const auto end = in.end();
  c.insert(in.begin(), in.end());
  benchmark::DoNotOptimize(c);
  benchmark::DoNotOptimize(in);
  while (st.KeepRunning()) {
    for (auto it = in.begin(); it != end; ++it) {
      benchmark::DoNotOptimize(&(*c.insert(*it).first));
    }
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_EmplaceDuplicate(benchmark::State& st, Container c, GenInputs gen) {
  auto in        = gen(st.range(0));
  const auto end = in.end();
  c.insert(in.begin(), in.end());
  benchmark::DoNotOptimize(c);
  benchmark::DoNotOptimize(in);
  while (st.KeepRunning()) {
    for (auto it = in.begin(); it != end; ++it) {
      benchmark::DoNotOptimize(&(*c.emplace(*it).first));
    }
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_Find(benchmark::State& st, Container c, GenInputs gen) {
  auto in = gen(st.range(0));
  c.insert(in.begin(), in.end());
  benchmark::DoNotOptimize(&(*c.begin()));
  const auto end = in.data() + in.size();
  while (st.KeepRunning()) {
    for (auto it = in.data(); it != end; ++it) {
      benchmark::DoNotOptimize(&(*c.find(*it)));
    }
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_FindRehash(benchmark::State& st, Container c, GenInputs gen) {
  c.rehash(8);
  auto in = gen(st.range(0));
  c.insert(in.begin(), in.end());
  benchmark::DoNotOptimize(&(*c.begin()));
  const auto end = in.data() + in.size();
  while (st.KeepRunning()) {
    for (auto it = in.data(); it != end; ++it) {
      benchmark::DoNotOptimize(&(*c.find(*it)));
    }
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_Rehash(benchmark::State& st, Container c, GenInputs gen) {
  auto in = gen(st.range(0));
  c.max_load_factor(3.0);
  c.insert(in.begin(), in.end());
  benchmark::DoNotOptimize(c);
  const auto bucket_count = c.bucket_count();
  while (st.KeepRunning()) {
    c.rehash(bucket_count + 1);
    c.rehash(bucket_count);
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_Compare_same_container(benchmark::State& st, Container, GenInputs gen) {
  auto in = gen(st.range(0));
  Container c1(in.begin(), in.end());
  Container c2 = c1;

  benchmark::DoNotOptimize(&(*c1.begin()));
  benchmark::DoNotOptimize(&(*c2.begin()));
  while (st.KeepRunning()) {
    bool res = c1 == c2;
    benchmark::DoNotOptimize(&res);
    benchmark::ClobberMemory();
  }
}

template <class Container, class GenInputs>
void BM_Compare_different_containers(benchmark::State& st, Container, GenInputs gen) {
  auto in1 = gen(st.range(0));
  auto in2 = gen(st.range(0));
  Container c1(in1.begin(), in1.end());
  Container c2(in2.begin(), in2.end());

  benchmark::DoNotOptimize(&(*c1.begin()));
  benchmark::DoNotOptimize(&(*c2.begin()));
  while (st.KeepRunning()) {
    bool res = c1 == c2;
    benchmark::DoNotOptimize(&res);
    benchmark::ClobberMemory();
  }
}

} // namespace ContainerBenchmarks

#endif // TEST_BENCHMARKS_CONTAINERS_CONTAINER_BENCHMARKS_H
