//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "benchmark/benchmark.h"
#include "test_macros.h"

constexpr std::size_t MAX_STRING_LEN = 8 << 14;

// No match found -- worst case for find_first_of (must scan entire haystack).
// Varies haystack size with a fixed needle size.
static void BM_StringFindFirstOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("xyz!@#$%");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfNoMatch)->Range(32, MAX_STRING_LEN);

// No match found -- varies needle size with a fixed haystack.
// Demonstrates O(n*m) vs O(n+m) scaling with needle size.
static void BM_StringFindFirstOfNoMatchVaryNeedle(benchmark::State& state) {
  std::string haystack(8192, 'a');
  std::string needle(state.range(0), 'b');
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfNoMatchVaryNeedle)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);

// Match at the end -- must scan nearly the entire haystack before finding it.
static void BM_StringFindFirstOfMatchAtEnd(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  haystack.back() = 'z';
  std::string needle("xyz!@#$%");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfMatchAtEnd)->Range(32, MAX_STRING_LEN);

// find_last_of -- no match, must scan entire haystack backward.
static void BM_StringFindLastOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("xyz!@#$%");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_last_of(needle));
  }
}
BENCHMARK(BM_StringFindLastOfNoMatch)->Range(32, MAX_STRING_LEN);

// find_first_not_of -- all characters are in the needle, no "not of" found.
static void BM_StringFindFirstNotOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("abcdefgh");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_not_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstNotOfNoMatch)->Range(32, MAX_STRING_LEN);

// find_last_not_of -- all characters are in the needle, no "not of" found.
static void BM_StringFindLastNotOfNoMatch(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle("abcdefgh");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_last_not_of(needle));
  }
}
BENCHMARK(BM_StringFindLastNotOfNoMatch)->Range(32, MAX_STRING_LEN);

// The benchmarks below cover cases where building a lookup structure could
// make the call slower than a plain scan.

// Early match: the match is at index state.range(0) of a 1024-character
// haystack, e.g. find_first_not_of(" \t\n") in trimming code.
static void BM_StringFindFirstNotOfEarlyMatch(benchmark::State& state) {
  std::string haystack(1024, 'x');
  for (std::int64_t i = 0; i < state.range(0); ++i)
    haystack[i] = ' ';
  std::string needle(" \t\n");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_not_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstNotOfEarlyMatch)->Arg(0)->Arg(3);

static void BM_StringFindFirstOfEarlyMatch(benchmark::State& state) {
  std::string haystack(1024, 'a');
  haystack[state.range(0)] = 'z';
  std::string needle("xyz");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfEarlyMatch)->Arg(0)->Arg(3);

// Early match from the back: the match is state.range(0) characters before the
// end of a 1024-character haystack.
static void BM_StringFindLastOfEarlyMatch(benchmark::State& state) {
  std::string haystack(1024, 'a');
  haystack[haystack.size() - 1 - state.range(0)] = 'z';
  std::string needle("xyz");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_last_of(needle));
  }
}
BENCHMARK(BM_StringFindLastOfEarlyMatch)->Arg(0)->Arg(3);

static void BM_StringFindLastNotOfEarlyMatch(benchmark::State& state) {
  std::string haystack(1024, 'x');
  for (std::int64_t i = 0; i < state.range(0); ++i)
    haystack[haystack.size() - 1 - i] = ' ';
  std::string needle(" \t\n");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_last_not_of(needle));
  }
}
BENCHMARK(BM_StringFindLastNotOfEarlyMatch)->Arg(0)->Arg(3);

// SSO-sized haystack with no match, varying the needle size.
static void BM_StringFindFirstOfSSO(benchmark::State& state) {
  std::string haystack(20, 'a');
  std::string needle(state.range(0), 'b');
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfSSO)->Arg(1)->Arg(2)->Arg(3)->Arg(8);

static void BM_StringFindLastNotOfSSO(benchmark::State& state) {
  std::string haystack(20, 'a');
  std::string needle(state.range(0), 'a');
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_last_not_of(needle));
  }
}
BENCHMARK(BM_StringFindLastNotOfSSO)->Arg(1)->Arg(2)->Arg(3)->Arg(8);

// Small needles on a long haystack with no match. A 1-character needle should
// perform like find(c).
static void BM_StringFindFirstOfSmallNeedle(benchmark::State& state) {
  std::string haystack(8192, 'a');
  std::string needle(state.range(0), 'b');
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfSmallNeedle)->Arg(1)->Arg(2);

static void BM_StringFindChar(benchmark::State& state) {
  std::string haystack(8192, 'a');
  char c = 'b';
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(c);
    benchmark::DoNotOptimize(haystack.find(c));
  }
}
BENCHMARK(BM_StringFindChar);

// Match just past the first few characters, so that the cost of building the
// lookup structure is not amortized over a long scan.
static void BM_StringFindFirstOfShortScan(benchmark::State& state) {
  std::string haystack(1024, 'a');
  haystack[state.range(0)] = 'z';
  std::string needle("xyz!@#$%");
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfShortScan)->Arg(17)->Arg(24)->Arg(32);

// Needle longer than the haystack, so building a lookup structure can't pay off.
static void BM_StringFindFirstOfLongNeedleShortHaystack(benchmark::State& state) {
  std::string haystack(state.range(0), 'a');
  std::string needle(200, 'b');
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}
BENCHMARK(BM_StringFindFirstOfLongNeedleShortHaystack)->Arg(20)->Arg(40)->Arg(200);

// Wide character types. Characters below 256 are looked up in the set; larger
// ones are rejected immediately if the needle has none, and otherwise searched
// for in the needle.
template <class CharT>
static std::basic_string<CharT> make_wide_text(std::size_t size, bool ascii, bool cjk) {
  std::basic_string<CharT> text;
  for (std::size_t i = 0; i != size; ++i) {
    // Mixed text alternates between runs of 3 ASCII and 3 CJK characters.
    bool use_cjk = cjk && (!ascii || (i / 3) % 2 == 1);
    text.push_back(use_cjk ? static_cast<CharT>(0x4E00 + i % 64) : static_cast<CharT>('a' + i % 26));
  }
  return text;
}

template <class CharT>
static void
BM_WideFindFirstOf(benchmark::State& state, bool ascii_text, bool cjk_text, std::basic_string<CharT> needle) {
  std::basic_string<CharT> haystack = make_wide_text<CharT>(state.range(0), ascii_text, cjk_text);
  for (auto _ : state) {
    benchmark::DoNotOptimize(haystack);
    benchmark::DoNotOptimize(needle);
    benchmark::DoNotOptimize(haystack.find_first_of(needle));
  }
}

template <class CharT>
static std::basic_string<CharT> make_wide_needle(std::initializer_list<char32_t> chars) {
  std::basic_string<CharT> needle;
  for (char32_t c : chars)
    needle.push_back(static_cast<CharT>(c));
  return needle;
}

// None of the needles occur in the generated text, so each call scans the whole
// haystack.
template <class CharT>
static void register_wide_find_first_of_benchmarks(const char* type_name) {
  struct Case {
    const char* name;
    bool ascii_text;
    bool cjk_text;
    std::basic_string<CharT> needle;
  };
  const Case cases[] = {
      {"ascii_needle_ascii_text",
       true,
       false,
       make_wide_needle<CharT>({' ', '\t', '\r', '\n', ',', ';', '.', '!', '?'})},
      {"ascii_needle_cjk_text", false, true, make_wide_needle<CharT>({' ', '\t', '\r', '\n', ',', ';', '.', '!', '?'})},
      {"mixed_needle_mixed_text",
       true,
       true,
       make_wide_needle<CharT>({' ', '\t', '\r', '\n', ',', ';', 0x3001, 0x3002})},
      {"cjk_needle_cjk_text", false, true, make_wide_needle<CharT>({0x3000, 0x3001, 0x3002, 0xFF0C})},
  };
  for (const Case& c : cases)
    benchmark::RegisterBenchmark(
        std::string("BM_WideFindFirstOf<") + type_name + ">/" + c.name,
        [c](benchmark::State& state) { BM_WideFindFirstOf<CharT>(state, c.ascii_text, c.cjk_text, c.needle); })
        ->Range(32, MAX_STRING_LEN);
}

int main(int argc, char** argv) {
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  register_wide_find_first_of_benchmarks<wchar_t>("wchar_t");
#endif
  register_wide_find_first_of_benchmarks<char16_t>("char16_t");
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
