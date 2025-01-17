//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_BENCHMARKS_CONTAINERS_ASSOCIATIVE_CONTAINER_BENCHMARKS_H
#define TEST_BENCHMARKS_CONTAINERS_ASSOCIATIVE_CONTAINER_BENCHMARKS_H

#include <algorithm>
#include <iterator>
#include <map>
#include <flat_map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "../GenerateInput.h"
#include "test_macros.h"

namespace support {

template <class Container>
struct adapt_operations;

template <class K>
struct adapt_operations<std::set<K>> {
  using ValueType = typename std::set<K>::value_type;
  using KeyType   = typename std::set<K>::key_type;
  static ValueType value_from_key(KeyType const& k) { return k; }
  static KeyType key_from_value(ValueType const& value) { return value; }
};

template <class K, class V>
struct adapt_operations<std::map<K, V>> {
  using ValueType = typename std::map<K, V>::value_type;
  using KeyType   = typename std::map<K, V>::key_type;
  static ValueType value_from_key(KeyType const& k) { return {k, Generate<V>::arbitrary()}; }
  static KeyType key_from_value(ValueType const& value) { return value.first; }
};

#if TEST_STD_VER >= 26
template <class K, class V>
struct adapt_operations<std::flat_map<K, V>> {
  using ValueType = typename std::map<K, V>::value_type;
  using KeyType   = typename std::map<K, V>::key_type;
  static ValueType value_from_key(KeyType const& k) { return {k, Generate<V>::arbitrary()}; }
  static KeyType key_from_value(ValueType const& value) { return value.first; }
};
#endif

template <class Container>
void associative_container_benchmarks(std::string container) {
  using Key   = typename Container::key_type;
  using Value = typename Container::value_type;

  auto generate_unique_keys = [=](std::size_t n) {
    std::set<Key> keys;
    while (keys.size() < n) {
      Key k = Generate<Key>::random();
      keys.insert(k);
    }
    return std::vector<Key>(keys.begin(), keys.end());
  };

  auto add_dummy_mapped_type = [](std::vector<Key> const& keys) {
    std::vector<Value> kv;
    for (Key const& k : keys)
      kv.push_back(adapt_operations<Container>::value_from_key(k));
    return kv;
  };

  auto get_key = [](Value const& v) { return adapt_operations<Container>::key_from_value(v); };

  // These benchmarks are structured to perform the operation being benchmarked
  // a small number of times at each iteration, in order to offset the cost of
  // PauseTiming() and ResumeTiming().
  static constexpr std::size_t BatchSize = 10;

  struct ScratchSpace {
    char storage[sizeof(Container)];
  };

  /////////////////////////
  // Constructors
  /////////////////////////
  benchmark::RegisterBenchmark(container + "::ctor(const&)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
    Container src(in.begin(), in.end());
    ScratchSpace c[BatchSize];

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        new (c + i) Container(src);
        benchmark::DoNotOptimize(c + i);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        reinterpret_cast<Container*>(c + i)->~Container();
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::ctor(iterator, iterator) (unsorted sequence)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::mt19937 randomness;
    std::vector<Key> keys = generate_unique_keys(size);
    std::shuffle(keys.begin(), keys.end(), randomness);
    std::vector<Value> in = add_dummy_mapped_type(keys);
    ScratchSpace c[BatchSize];

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        new (c + i) Container(in.begin(), in.end());
        benchmark::DoNotOptimize(c + i);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        reinterpret_cast<Container*>(c + i)->~Container();
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::ctor(iterator, iterator) (sorted sequence)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Key> keys  = generate_unique_keys(size);
    std::sort(keys.begin(), keys.end());
    std::vector<Value> in = add_dummy_mapped_type(keys);
    ScratchSpace c[BatchSize];

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        new (c + i) Container(in.begin(), in.end());
        benchmark::DoNotOptimize(c + i);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        reinterpret_cast<Container*>(c + i)->~Container();
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  /////////////////////////
  // Assignment
  /////////////////////////
  benchmark::RegisterBenchmark(container + "::operator=(const&)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
    Container src(in.begin(), in.end());
    Container c[BatchSize];

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i] = src;
        benchmark::DoNotOptimize(c[i]);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].clear();
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  /////////////////////////
  // Insertion
  /////////////////////////
  benchmark::RegisterBenchmark(container + "::insert(value) (already present)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
    Value to_insert        = in[in.size() / 2]; // pick any existing value
    std::vector<Container> c(BatchSize, Container(in.begin(), in.end()));

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].insert(to_insert);
        benchmark::DoNotOptimize(c[i]);
        benchmark::ClobberMemory();
      }

      // There is no cleanup to do, since associative containers don't insert
      // if the key is already present.
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::insert(value) (new value)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size + 1));
    Value to_insert        = in.back();
    in.pop_back();
    std::vector<Container> c(BatchSize, Container(in.begin(), in.end()));

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].insert(to_insert);
        benchmark::DoNotOptimize(c[i]);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].erase(get_key(to_insert));
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::insert(hint, value) (good hint)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size + 1));
    Value to_insert        = in.back();
    in.pop_back();

    std::vector<Container> c(BatchSize, Container(in.begin(), in.end()));
    typename Container::iterator hints[BatchSize];
    for (std::size_t i = 0; i != BatchSize; ++i) {
      hints[i] = c[i].lower_bound(get_key(to_insert));
    }

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].insert(hints[i], to_insert);
        benchmark::DoNotOptimize(c[i]);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].erase(get_key(to_insert));
        hints[i] = c[i].lower_bound(get_key(to_insert)); // refresh hints in case of invalidation
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::insert(hint, value) (bad hint)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size + 1));
    Value to_insert        = in.back();
    in.pop_back();
    std::vector<Container> c(BatchSize, Container(in.begin(), in.end()));

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].insert(c[i].begin(), to_insert);
        benchmark::DoNotOptimize(c[i]);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].erase(get_key(to_insert));
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::insert(iterator, iterator) (all new keys)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size + (size / 10)));

    // Populate a container with a small number of elements, that's what containers will start with.
    std::vector<Value> small;
    for (std::size_t i = 0; i != (size / 10); ++i) {
      small.push_back(in.back());
      in.pop_back();
    }
    Container c(small.begin(), small.end());

    for (auto _ : st) {
      c.insert(in.begin(), in.end());
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();

      st.PauseTiming();
      c = Container(small.begin(), small.end());
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::insert(iterator, iterator) (half new keys)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));

    // Populate a container that already contains half the elements we'll try inserting,
    // that's what our container will start with.
    std::vector<Value> small;
    for (std::size_t i = 0; i != size / 2; ++i) {
      small.push_back(in.at(i * 2));
    }
    Container c(small.begin(), small.end());

    for (auto _ : st) {
      c.insert(in.begin(), in.end());
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();

      st.PauseTiming();
      c = Container(small.begin(), small.end());
      st.ResumeTiming();
    }
  })->Arg(1024);

  /////////////////////////
  // Erasure
  /////////////////////////
  benchmark::RegisterBenchmark(container + "::erase(key) (existent)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
    Value element          = in[in.size() / 2]; // pick any element
    std::vector<Container> c(BatchSize, Container(in.begin(), in.end()));

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].erase(get_key(element));
        benchmark::DoNotOptimize(c[i]);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].insert(element);
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::erase(key) (non-existent)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size + 1));
    Value element          = in.back();
    in.pop_back();
    Container c(in.begin(), in.end());

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c.erase(get_key(element));
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
      }

      // no cleanup required because we erased a non-existent element
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::erase(iterator)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
    Value element          = in[in.size() / 2]; // pick any element

    std::vector<Container> c;
    std::vector<typename Container::iterator> iterators;
    for (std::size_t i = 0; i != BatchSize; ++i) {
      c.push_back(Container(in.begin(), in.end()));
      iterators.push_back(c[i].find(get_key(element)));
    }

    while (st.KeepRunningBatch(BatchSize)) {
      for (std::size_t i = 0; i != BatchSize; ++i) {
        c[i].erase(iterators[i]);
        benchmark::DoNotOptimize(c[i]);
        benchmark::ClobberMemory();
      }

      st.PauseTiming();
      for (std::size_t i = 0; i != BatchSize; ++i) {
        iterators[i] = c[i].insert(element).first;
      }
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::erase(iterator, iterator) (erase half the container)", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
    Container c(in.begin(), in.end());

    auto first = std::next(c.begin(), c.size() / 4);
    auto last  = std::next(c.begin(), 3 * (c.size() / 4));
    for (auto _ : st) {
      c.erase(first, last);
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();

      st.PauseTiming();
      c     = Container(in.begin(), in.end());
      first = std::next(c.begin(), c.size() / 4);
      last  = std::next(c.begin(), 3 * (c.size() / 4));
      st.ResumeTiming();
    }
  })->Arg(1024);

  benchmark::RegisterBenchmark(container + "::clear()", [=](auto& st) {
    const std::size_t size = st.range(0);
    std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
    Container c(in.begin(), in.end());

    for (auto _ : st) {
      c.clear();
      benchmark::DoNotOptimize(c);
      benchmark::ClobberMemory();

      st.PauseTiming();
      c = Container(in.begin(), in.end());
      st.ResumeTiming();
    }
  })->Arg(1024);

  /////////////////////////
  // Query
  /////////////////////////
  auto bench_with_existent_key = [=](auto func) {
    return [=](auto& st) {
      const std::size_t size = st.range(0);
      std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size));
      Value element          = in[in.size() / 2]; // pick any element
      Container c(in.begin(), in.end());

      while (st.KeepRunningBatch(BatchSize)) {
        for (std::size_t i = 0; i != BatchSize; ++i) {
          auto result = func(c, element);
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(result);
          benchmark::ClobberMemory();
        }
      }
    };
  };

  auto bench_with_nonexistent_key = [=](auto func) {
    return [=](auto& st) {
      const std::size_t size = st.range(0);
      std::vector<Value> in  = add_dummy_mapped_type(generate_unique_keys(size + 1));
      Value element          = in.back();
      in.pop_back();
      Container c(in.begin(), in.end());

      while (st.KeepRunningBatch(BatchSize)) {
        for (std::size_t i = 0; i != BatchSize; ++i) {
          auto result = func(c, element);
          benchmark::DoNotOptimize(c);
          benchmark::DoNotOptimize(result);
          benchmark::ClobberMemory();
        }
      }
    };
  };

  benchmark::RegisterBenchmark(
      container + "::find(key) (existent)",
      bench_with_existent_key([=](Container const& c, Value const& element) { return c.find(get_key(element)); }))
      ->Arg(1024);
  benchmark::RegisterBenchmark(
      container + "::find(key) (non-existent)",
      bench_with_nonexistent_key([=](Container const& c, Value const& element) { return c.find(get_key(element)); }))
      ->Arg(1024);

  benchmark::RegisterBenchmark(
      container + "::count(key) (existent)",
      bench_with_existent_key([=](Container const& c, Value const& element) { return c.count(get_key(element)); }))
      ->Arg(1024);
  benchmark::RegisterBenchmark(
      container + "::count(key) (non-existent)",
      bench_with_nonexistent_key([=](Container const& c, Value const& element) { return c.count(get_key(element)); }))
      ->Arg(1024);

  benchmark::RegisterBenchmark(
      container + "::contains(key) (existent)",
      bench_with_existent_key([=](Container const& c, Value const& element) { return c.contains(get_key(element)); }))
      ->Arg(1024);
  benchmark::RegisterBenchmark(
      container + "::contains(key) (non-existent)",
      bench_with_nonexistent_key([=](Container const& c, Value const& element) {
        return c.contains(get_key(element));
      }))
      ->Arg(1024);

  benchmark::RegisterBenchmark(
      container + "::lower_bound(key) (existent)",
      bench_with_existent_key([=](Container const& c, Value const& element) {
        return c.lower_bound(get_key(element));
      }))
      ->Arg(1024);
  benchmark::RegisterBenchmark(
      container + "::lower_bound(key) (non-existent)",
      bench_with_nonexistent_key([=](Container const& c, Value const& element) {
        return c.lower_bound(get_key(element));
      }))
      ->Arg(1024);

  benchmark::RegisterBenchmark(
      container + "::upper_bound(key) (existent)",
      bench_with_existent_key([=](Container const& c, Value const& element) {
        return c.upper_bound(get_key(element));
      }))
      ->Arg(1024);
  benchmark::RegisterBenchmark(
      container + "::upper_bound(key) (non-existent)",
      bench_with_nonexistent_key([=](Container const& c, Value const& element) {
        return c.upper_bound(get_key(element));
      }))
      ->Arg(1024);

  benchmark::RegisterBenchmark(
      container + "::equal_range(key) (existent)",
      bench_with_existent_key([=](Container const& c, Value const& element) {
        return c.equal_range(get_key(element));
      }))
      ->Arg(1024);
  benchmark::RegisterBenchmark(
      container + "::equal_range(key) (non-existent)",
      bench_with_nonexistent_key([=](Container const& c, Value const& element) {
        return c.equal_range(get_key(element));
      }))
      ->Arg(1024);
}

} // namespace support

#endif // TEST_BENCHMARKS_CONTAINERS_ASSOCIATIVE_CONTAINER_BENCHMARKS_H
