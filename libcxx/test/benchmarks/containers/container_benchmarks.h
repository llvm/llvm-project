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
  char buffer[sizeof(Container)];
  for (auto _ : st) {
    std::construct_at(reinterpret_cast<Container*>(buffer), size);
    benchmark::DoNotOptimize(buffer);
    st.PauseTiming();
    std::destroy_at(reinterpret_cast<Container*>(buffer));
    st.ResumeTiming();
  }
}

template <class Container, class Generator>
void BM_ctor_size_value(benchmark::State& st, Generator gen) {
  using ValueType = typename Container::value_type;
  const auto size = st.range(0);
  ValueType value = gen();
  benchmark::DoNotOptimize(value);
  char buffer[sizeof(Container)];
  for (auto _ : st) {
    std::construct_at(reinterpret_cast<Container*>(buffer), size, value);
    benchmark::DoNotOptimize(buffer);
    st.PauseTiming();
    std::destroy_at(reinterpret_cast<Container*>(buffer));
    st.ResumeTiming();
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
  char buffer[sizeof(Container)];
  for (auto _ : st) {
    std::construct_at(reinterpret_cast<Container*>(buffer), begin, end);
    benchmark::DoNotOptimize(buffer);
    st.PauseTiming();
    std::destroy_at(reinterpret_cast<Container*>(buffer));
    st.ResumeTiming();
  }
}

#if TEST_STD_VER >= 23
template <class Container>
void BM_ctor_from_range(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const auto size = st.range(0);
  std::vector<ValueType> in(size);
  benchmark::DoNotOptimize(in);
  char buffer[sizeof(Container)];
  for (auto _ : st) {
    std::construct_at(reinterpret_cast<Container*>(buffer), std::from_range, in);
    benchmark::DoNotOptimize(buffer);
    st.PauseTiming();
    std::destroy_at(reinterpret_cast<Container*>(buffer));
    st.ResumeTiming();
  }
}
#endif

template <class Container>
void BM_ctor_copy(benchmark::State& st) {
  auto size = st.range(0);
  Container c(size);
  char buffer[sizeof(Container)];
  for (auto _ : st) {
    std::construct_at(reinterpret_cast<Container*>(buffer), c);
    benchmark::DoNotOptimize(buffer);
    st.PauseTiming();
    std::destroy_at(reinterpret_cast<Container*>(buffer));
    st.ResumeTiming();
  }
}

template <class Container>
void BM_assignment(benchmark::State& st) {
  auto size = st.range(0);
  Container c1;
  Container c2(size);
  for (auto _ : st) {
    c1 = c2;
    DoNotOptimizeData(c1);
    DoNotOptimizeData(c2);
  }
}

template <typename Container>
void BM_assign_inputiter(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  auto size       = st.range(0);
  std::vector<ValueType> inputs(size);
  Container c(inputs.begin(), inputs.end());
  DoNotOptimizeData(c);
  DoNotOptimizeData(inputs);
  ValueType* first = inputs.data();
  ValueType* last  = inputs.data() + inputs.size();

  for (auto _ : st) {
    c.assign(cpp17_input_iterator(first), cpp17_input_iterator(last));
    benchmark::ClobberMemory();
  }
}

template <class Container>
void BM_insert_start(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  Container c(inputs.begin(), inputs.end());
  DoNotOptimizeData(c);

  ValueType value{};
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    c.insert(c.begin(), value);
    DoNotOptimizeData(c);

    c.erase(std::prev(c.end())); // avoid growing indefinitely
  }
}

template <class Container>
  requires std::random_access_iterator<typename Container::iterator>
void BM_insert_middle(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  Container c(inputs.begin(), inputs.end());
  DoNotOptimizeData(c);

  ValueType value{};
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    auto mid = c.begin() + (count / 2); // requires random-access iterators in order to make sense
    c.insert(mid, value);
    DoNotOptimizeData(c);

    c.erase(c.end() - 1); // avoid growing indefinitely
  }
}

template <class Container>
void BM_insert_input_iter_with_reserve_no_realloc(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  const auto beg = cpp17_input_iterator(inputs.begin());
  const auto end = cpp17_input_iterator(inputs.end());

  auto size = 100; // arbitrary
  Container c(size);
  c.reserve(size + inputs.size()); // ensure no reallocation
  for (auto _ : st) {
    c.insert(c.begin(), beg, end);
    DoNotOptimizeData(c);

    st.PauseTiming();
    c.erase(c.begin() + size, c.end()); // avoid growing indefinitely
    st.ResumeTiming();
  }
}

template <class Container>
void BM_insert_input_iter_with_reserve_half_filled(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  const auto beg = cpp17_input_iterator(inputs.begin());
  const auto end = cpp17_input_iterator(inputs.end());

  for (auto _ : st) {
    st.PauseTiming();
    Container c(count / 2);
    // Half the elements in [beg, end) can fit in the vector without reallocation, so we'll reallocate halfway through
    c.reserve(count);
    st.ResumeTiming();

    c.insert(c.begin(), beg, end);
    DoNotOptimizeData(c);
  }
}

template <class Container>
void BM_insert_input_iter_with_reserve_near_full(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  const auto beg = cpp17_input_iterator(inputs.begin());
  const auto end = cpp17_input_iterator(inputs.end());

  for (auto _ : st) {
    st.PauseTiming();
    Container c(count);
    c.reserve(count + 5); // Make sure the container is almost full
    st.ResumeTiming();

    c.insert(c.begin(), beg, end);
    DoNotOptimizeData(c);
  }
}

template <class Container>
void BM_erase_start(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  Container c(inputs.begin(), inputs.end());
  DoNotOptimizeData(c);

  ValueType value{};
  benchmark::DoNotOptimize(value);
  for (auto _ : st) {
    c.erase(c.begin());
    DoNotOptimizeData(c);

    c.insert(c.end(), value); // re-insert an element at the end to avoid needing a new container
  }
}

template <class Container>
  requires std::random_access_iterator<typename Container::iterator>
void BM_erase_middle(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  Container c(inputs.begin(), inputs.end());
  DoNotOptimizeData(c);

  ValueType value{};
  benchmark::DoNotOptimize(value);

  for (auto _ : st) {
    auto mid = c.begin() + (count / 2);
    c.erase(mid);
    DoNotOptimizeData(c);

    c.insert(c.end(), value); // re-insert an element at the end to avoid needing a new container
  }
}

template <class Container>
void sequence_container_benchmarks(std::string container) {
  using ValueType = typename Container::value_type;
  auto cheap      = [] { return Generate<ValueType>::cheap(); };
  auto expensive  = [] { return Generate<ValueType>::expensive(); };

  // constructors
  benchmark::RegisterBenchmark(container + "::ctor(size)", BM_ctor_size<Container>)->Arg(1024);
  benchmark::RegisterBenchmark(container + "::ctor(size, value_type) (cheap elements)", [=](auto& st) {
    BM_ctor_size_value<Container>(st, cheap);
  })->Arg(1024);
  benchmark::RegisterBenchmark(container + "::ctor(size, value_type) (expensive elements)", [=](auto& st) {
    BM_ctor_size_value<Container>(st, expensive);
  })->Arg(1024);
  benchmark::RegisterBenchmark(container + "::ctor(Iterator, Iterator) (cheap elements)", [=](auto& st) {
    BM_ctor_iter_iter<Container>(st, cheap);
  })->Arg(1024);
  benchmark::RegisterBenchmark(container + "::ctor(Iterator, Iterator) (expensive elements)", [=](auto& st) {
    BM_ctor_iter_iter<Container>(st, expensive);
  })->Arg(1024);
#if TEST_STD_VER >= 23
  benchmark::RegisterBenchmark(container + "::ctor(Range)", BM_ctor_from_range<Container>)->Arg(1024);
#endif
  benchmark::RegisterBenchmark(container + "::ctor(const&)", BM_ctor_copy<Container>)->Arg(1024);

  // assignment
  benchmark::RegisterBenchmark(container + "::operator=", BM_assignment<Container>)->Arg(1024);
  benchmark::RegisterBenchmark(container + "::assign(input-iter, input-iter)", BM_assign_inputiter<Container>)
      ->Arg(1024);

  // insert
  benchmark::RegisterBenchmark(container + "::insert(start)", BM_insert_start<Container>)->Arg(1024);
  if constexpr (std::random_access_iterator<typename Container::iterator>) {
    benchmark::RegisterBenchmark(container + "::insert(middle)", BM_insert_middle<Container>)->Arg(1024);
  }
  if constexpr (requires(Container c) { c.reserve(0); }) {
    benchmark::RegisterBenchmark(container + "::insert(input-iter, input-iter) (no realloc)",
                                 BM_insert_input_iter_with_reserve_no_realloc<Container>)
        ->Arg(514048);
    benchmark::RegisterBenchmark(container + "::insert(input-iter, input-iter) (half filled)",
                                 BM_insert_input_iter_with_reserve_no_realloc<Container>)
        ->Arg(514048);
    benchmark::RegisterBenchmark(container + "::insert(input-iter, input-iter) (near full)",
                                 BM_insert_input_iter_with_reserve_near_full<Container>)
        ->Arg(514048);
  }

  // erase
  benchmark::RegisterBenchmark(container + "::erase(start)", BM_erase_start<Container>)->Arg(1024);
  if constexpr (std::random_access_iterator<typename Container::iterator>) {
    benchmark::RegisterBenchmark(container + "::erase(middle)", BM_erase_middle<Container>)->Arg(1024);
  }
}

//
// "Back-insertable" sequence container operations
//
template <class Container>
void BM_push_back(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  benchmark::DoNotOptimize(inputs);

  Container c;
  DoNotOptimizeData(c);
  while (st.KeepRunningBatch(count)) {
    c.clear();
    for (int i = 0; i != count; ++i) {
      c.push_back(inputs[i]);
    }
    DoNotOptimizeData(c);
  }
}

template <class Container>
void BM_push_back_with_reserve(benchmark::State& st) {
  using ValueType = typename Container::value_type;
  const int count = st.range(0);
  std::vector<ValueType> inputs(count);
  benchmark::DoNotOptimize(inputs);

  Container c;
  c.reserve(count);
  DoNotOptimizeData(c);
  while (st.KeepRunningBatch(count)) {
    c.clear();
    for (int i = 0; i != count; ++i) {
      c.push_back(inputs[i]);
    }
    DoNotOptimizeData(c);
  }
}

template <class Container>
void back_insertable_container_benchmarks(std::string container) {
  sequence_container_benchmarks<Container>(container);
  benchmark::RegisterBenchmark(container + "::push_back()", BM_push_back<Container>)->Arg(1024);
  if constexpr (requires(Container c) { c.reserve(0); }) {
    benchmark::RegisterBenchmark(container + "::push_back() (with reserve)", BM_push_back_with_reserve<Container>)
        ->Arg(1024);
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
