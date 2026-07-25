//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "make_string.h"
#include "test_macros.h"

constexpr size_t small_size = 5;
constexpr size_t large_size = 30;

std::string rename(std::string str, std::string_view replacement) {
  while (true) {
    auto pos = str.find("basic_string");
    if (pos == std::string::npos)
      return str;
    str.replace(pos, std::strlen("basic_string"), replacement);
  }
}

template <class Func, class Mod = decltype([](auto) {})>
void bench(std::string name, Func func, Mod modifier = {}) {
  benchmark::RegisterBenchmark(rename(name, "string"), [=](benchmark::State& state) {
    func(std::type_identity<char>(), state);
  })->Apply(modifier);

  benchmark::RegisterBenchmark(rename(name, "u8string"), [=](benchmark::State& state) {
    func(std::type_identity<char8_t>(), state);
  })->Apply(modifier);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  benchmark::RegisterBenchmark(rename(name, "wstring"), [=](benchmark::State& state) {
    func(std::type_identity<wchar_t>(), state);
  })->Apply(modifier);
#endif
}

int main(int argc, char** argv) {
  // [string.cons]
  bench("std::basic_string::ctor()", []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
    for (auto _ : state) {
      std::basic_string<CharT> str;
      benchmark::DoNotOptimize(str);
    }
  });

  bench(
      "std::basic_string::ctor(const value_type*)",
      []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
        std::basic_string<CharT> str(state.range(), 'a');

        for (auto _ : state) {
          auto ptr = str.data();
          std::basic_string<CharT> copy(ptr);
          benchmark::DoNotOptimize(copy);
        }
      },
      [](auto bm) { bm->Arg(small_size)->Arg(large_size); });

  {
    static auto bench_impl =
        []<bool opaque, class CharT>(std::bool_constant<opaque>, std::type_identity<CharT>, benchmark::State& state) {
          std::basic_string<CharT> str(state.range(), 'a');

          for (auto _ : state) {
            auto copy = str;
            benchmark::DoNotOptimize(copy);
          }
        };
    bench("std::basic_string::ctor(const Self&) (opaque)", std::bind_front(bench_impl, std::true_type{}), [](auto bm) {
      bm->Arg(small_size)->Arg(large_size);
    });

    bench("std::basic_string::ctor(const Self&) (transparent)",
          std::bind_front(bench_impl, std::false_type{}),
          [](auto bm) { bm->Arg(small_size)->Arg(large_size); });
  }

  bench(
      "std::basic_string::ctor(Self&&)",
      []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
        using str = std::basic_string<CharT>;

        union U {
          U() {}
          ~U() {}

          struct {
            str s1;
            str s2;
          };
        } u;

        std::construct_at(&u.s1, state.range(), 'a');

        for (auto _ : state) {
          std::construct_at(&u.s2, std::move(u.s1));
          std::destroy_at(&u.s1);
          benchmark::DoNotOptimize(u);

          std::construct_at(&u.s1, std::move(u.s2));
          std::destroy_at(&u.s2);
          benchmark::DoNotOptimize(u);
        }

        std::destroy_at(&u.s1);
      },
      [](auto bm) { bm->Arg(small_size)->Arg(large_size); });

  {
    static auto bench_impl =
        []<bool opaque, class CharT>(std::bool_constant<opaque>, std::type_identity<CharT>, benchmark::State& state) {
          using str = std::basic_string<CharT>;
          str src(state.range(), 'a');

          str strings[4096];
          while (state.KeepRunningBatch(std::size(strings))) {
            state.PauseTiming();
            for (auto& string : strings)
              str().swap(string); // Make sure the strings are in the default constructed state
            state.ResumeTiming();
            benchmark::DoNotOptimize(strings);

            for (auto& string : strings) {
              if constexpr (opaque)
                benchmark::DoNotOptimize(src);
              string = src;
            }
          }
        };

    bench("std::basic_string::operator=(const Self&) (opaque)",
          std::bind_front(bench_impl, std::true_type{}),
          [](auto bm) { bm->Arg(small_size)->Arg(large_size); });

    bench("std::basic_string::operator=(const Self&) (transparent)",
          std::bind_front(bench_impl, std::false_type{}),
          [](auto bm) { bm->Arg(small_size)->Arg(large_size); });
  }

  {
    static auto bench_impl =
        []<size_t size, bool opaque, class CharT>(
            std::integral_constant<size_t, size>,
            std::bool_constant<opaque>,
            std::type_identity<CharT>,
            benchmark::State& state) {
          using str = std::basic_string<CharT>;

          static constexpr std::array<CharT, size> src = [] {
            std::array<CharT, size> ret;
            std::ranges::fill(ret, 'a');
            ret[size - 1] = '\0';
            return ret;
          }();

          str strings[4096];
          while (state.KeepRunningBatch(std::size(strings))) {
            state.PauseTiming();
            for (auto& string : strings)
              str().swap(string); // Make sure the strings are in the default constructed state
            state.ResumeTiming();
            benchmark::DoNotOptimize(strings);

            for (auto& string : strings) {
              auto ptr = src.data();
              if constexpr (opaque)
                benchmark::DoNotOptimize(ptr);
              string = ptr;
            }
          }
        };

    bench("std::basic_string::operator=(const value_type*) (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string::operator=(const value_type*) (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming

    bench("std::basic_string::operator=(const value_type*) (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string::operator=(const value_type*) (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming
  }

  // [string.capacity]
  bench("std::basic_string::size()", []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
    std::basic_string<CharT> str;
    for (auto _ : state) {
      benchmark::DoNotOptimize(str);
      benchmark::DoNotOptimize(str.size());
    }
  });

#if TEST_STD_VER >= 23
  bench("std::basic_string::resize_and_overwrite()",
        []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
          std::basic_string<CharT> str;
          for (auto _ : state) {
            benchmark::DoNotOptimize(str);
            str.resize_and_overwrite(10, [](CharT* ptr, size_t n) {
              std::fill_n(ptr, n, 'a');
              return n;
            });
            benchmark::DoNotOptimize(str);
            str.clear();
          }
        });
#endif

  // [string.modifiers]

  {
    static auto bench_impl =
        []<size_t end, bool opaque, class CharT>(
            std::integral_constant<size_t, end>,
            std::bool_constant<opaque>,
            std::type_identity<CharT>,
            benchmark::State& state) {
          std::basic_string<CharT> strings[4096];

          size_t size = state.range();
          size_t pos  = size / 2;
          auto npos   = end;
          while (state.KeepRunningBatch(std::size(strings))) {
            state.PauseTiming();
            for (auto& string : strings)
              string.resize(size, 'a');
            state.ResumeTiming();
            for (auto& string : strings) {
              if constexpr (opaque) {
                benchmark::DoNotOptimize(pos);
                benchmark::DoNotOptimize(npos);
              }
              string.erase(pos, npos);
            }
          }
        };
    bench("std::basic_string::erase() (to end of string, opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, std::string::npos>{}, std::true_type{}),
          [](auto bm) { bm->Arg(small_size)->Arg(large_size); });
    bench("std::basic_string::erase() (to end of string, transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, std::string::npos>{}, std::false_type{}),
          [](auto bm) { bm->Arg(small_size)->Arg(large_size); });
    bench("std::basic_string::erase() (in the middle, opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, 2>{}, std::true_type{}),
          [](auto bm) { bm->Arg(small_size)->Arg(large_size); });
    bench("std::basic_string::erase() (in the middle, transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, 2>{}, std::false_type{}),
          [](auto bm) { bm->Arg(small_size)->Arg(large_size); });
  }

  // [string.ops]
  bench("std::basic_string::data()", []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
    std::basic_string<CharT> str;
    for (auto _ : state) {
      benchmark::DoNotOptimize(str);
      benchmark::DoNotOptimize(str.data());
    }
  });

  auto search_sizes = [](auto bm) { bm->Arg(small_size)->Arg(large_size)->Arg(8192); };
  bench(
      "std::basic_string::find(std::basic_string) (no match)",
      []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
        std::basic_string<CharT> haystack(state.range(), 'a');
        std::basic_string<CharT> needle(8, 'b');

        for (auto _ : state) {
          benchmark::DoNotOptimize(haystack.find(needle));
        }
      },
      search_sizes);

  bench(
      "std::basic_string::find(std::basic_string) (match at the end)",
      []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
        std::basic_string<CharT> haystack(state.range(), 'a');
        std::basic_string<CharT> needle(8, 'b');
        haystack += needle;

        for (auto _ : state) {
          benchmark::DoNotOptimize(haystack.find(needle));
        }
      },
      search_sizes);

  bench(
      "std::basic_string::find(const value_type*) (literal)",
      []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
        std::basic_string<CharT> s1(state.range(), 'a');

        for (auto _ : state) {
          benchmark::DoNotOptimize(s1.find(MAKE_CSTRING(CharT, "b")));
        }
      },
      search_sizes);

  bench(
      "std::basic_string::find(value_type) (literal)",
      []<class CharT>(std::type_identity<CharT>, benchmark::State& state) {
        std::basic_string<CharT> s1(state.range(), 'a');

        for (auto _ : state) {
          benchmark::DoNotOptimize(s1.find('b'));
        }
      },
      search_sizes);

  // [string.cmp]
  {
    static auto bench_impl =
        []<size_t size, bool opaque, class CharT>(
            std::integral_constant<size_t, size>,
            std::bool_constant<opaque>,
            std::type_identity<CharT>,
            benchmark::State& state) {
          using str = std::basic_string<CharT>;

          static constexpr std::array<CharT, size> src = [] {
            std::array<CharT, size> ret;
            std::ranges::fill(ret, 'a');
            ret[size - 1] = '\0';
            return ret;
          }();

          str strings[4096];
          std::fill_n(strings, 4096, src.data());
          while (state.KeepRunningBatch(std::size(strings))) {
            benchmark::DoNotOptimize(strings);

            for (auto& string : strings) {
              auto ptr = src.data();
              if constexpr (opaque)
                benchmark::DoNotOptimize(ptr);
              benchmark::DoNotOptimize(string == ptr);
            }
          }
        };

    bench("std::basic_string == const CharT* (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string == const CharT* (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming

    bench("std::basic_string == const CharT* (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string == const CharT* (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming
  }

  {
    static auto bench_impl =
        []<size_t size, bool opaque, class CharT>(
            std::integral_constant<size_t, size>,
            std::bool_constant<opaque>,
            std::type_identity<CharT>,
            benchmark::State& state) {
          using str = std::basic_string<CharT>;

          static constexpr std::array<CharT, size> src = [] {
            std::array<CharT, size> ret;
            std::ranges::fill(ret, 'a');
            ret[size - 1] = '\0';
            return ret;
          }();

          str strings[4096];
          std::fill_n(strings, 4096, src.data());
          while (state.KeepRunningBatch(std::size(strings))) {
            benchmark::DoNotOptimize(strings);

            for (auto& string : strings) {
              auto ptr = src.data();
              if constexpr (opaque)
                benchmark::DoNotOptimize(ptr);
              benchmark::DoNotOptimize(string.compare(ptr));
            }
          }
        };

    bench("std::basic_string::compare(const CharT*) (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string::compare(const CharT*) (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming

    bench("std::basic_string::compare(const CharT*) (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string::compare(const CharT*) (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming
  }

  // [string.compare]
  {
    static auto bench_impl =
        []<size_t size, bool opaque, class CharT>(
            std::integral_constant<size_t, size>,
            std::bool_constant<opaque>,
            std::type_identity<CharT>,
            benchmark::State& state) {
          using str = std::basic_string<CharT>;

          std::basic_string<CharT> src(size, 'a');

          str strings[4096];
          std::fill_n(strings, 4096, src.data());
          while (state.KeepRunningBatch(std::size(strings))) {
            benchmark::DoNotOptimize(strings);

            for (auto& string : strings) {
              if constexpr (opaque)
                benchmark::DoNotOptimize(src);
              benchmark::DoNotOptimize(string == src);
            }
          }
        };

    bench("std::basic_string == std::basic_string (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string == std::basic_string (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming

    bench("std::basic_string == std::basic_string (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string == std::basic_string (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming
  }

  {
    static auto bench_impl =
        []<size_t size, bool opaque, class CharT>(
            std::integral_constant<size_t, size>,
            std::bool_constant<opaque>,
            std::type_identity<CharT>,
            benchmark::State& state) {
          std::basic_string<CharT> src(size, 'a');

          std::basic_string<CharT> strings[4096];
          std::fill_n(strings, 4096, src.data());
          while (state.KeepRunningBatch(std::size(strings))) {
            benchmark::DoNotOptimize(strings);

            for (auto& string : strings) {
              if constexpr (opaque)
                benchmark::DoNotOptimize(src);
              benchmark::DoNotOptimize(string.compare(src));
            }
          }
        };

    // These also effectively cover operator<, since operator< is just forwarding to compare.

    bench("std::basic_string::compare(const std::basic_string&) (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string::compare(const std::basic_string&) (opaque)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::true_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming

    bench("std::basic_string::compare(const std::basic_string&) (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, small_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(small_size); }); // for naming

    bench("std::basic_string::compare(const std::basic_string&) (transparent)",
          std::bind_front(bench_impl, std::integral_constant<size_t, large_size>{}, std::false_type{}),
          [](auto bm) { bm->Arg(large_size); }); // for naming
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
