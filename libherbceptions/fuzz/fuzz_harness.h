//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
Shared plumbing for the libherbceptions fuzz targets: a bounded output
collector plus a standalone replay driver used when the target is not
linked against a fuzzing engine (build with HERBCEPTIONS_FUZZ_ENGINE=ON
to get a -fsanitize=fuzzer binary instead; then any corpus directory /
single inputs work as usual).
*/
#include <herbceptions/error>

#include <cstddef>
#include <cstdint>

namespace herbceptions_fuzz {

inline constexpr ::std::size_t capture_capacity{8192};

struct capture_ctx {
  char buf[capture_capacity];
  ::std::size_t len;
};

inline void reset(capture_ctx &ctx) noexcept {
  ctx.len = 0;
  ctx.buf[0] = '\0';
}

// Collector: appends every reported byte, truncating at capacity. Safe to
// pass to any do_query_information.
inline void capture_cookfun(void *cookie, ::std::io_scatter_t const *v,
                            ::std::size_t n) noexcept {
  auto *ctx{static_cast<capture_ctx *>(cookie)};
  for (::std::size_t i = 0; i < n; ++i) {
    auto const *base{static_cast<char const *>(v[i].base)};
    for (::std::size_t j = 0; j < v[i].len; ++j) {
      if (ctx->len + 1u == capture_capacity) {
        return;
      }
      ctx->buf[ctx->len++] = base[j];
    }
  }
  ctx->buf[ctx->len] = '\0';
}

} // namespace herbceptions_fuzz

#ifndef HERBCEPTIONS_HAVE_FUZZ_ENGINE

/*
Standalone driver: without arguments it replays stdin once. Otherwise:

    prog <iterations> [seed files...]

loads every seed file and then runs a self-contained mutation loop
(bit flips, byte replacements, insertions, splices, truncations) over the
seeds through LLVMFuzzerTestOneInput. A crash is a finding; rerun with the
saved input to reproduce it.
*/
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(::std::uint8_t const *data,
                                      ::std::size_t size);

namespace herbceptions_fuzz {

inline void mutate(::std::vector<::std::uint8_t> &v,
                   ::std::mt19937_64 &rng) noexcept {
  ::std::size_t const rounds{1u + rng() % 8u};
  for (::std::size_t r = 0; r != rounds && !v.empty(); ++r) {
    switch (rng() % 6u) {
    case 0: { // bit flip
      v[rng() % v.size()] ^= static_cast<::std::uint8_t>(1u << (rng() % 8u));
      break;
    }
    case 1: { // random byte
      v[rng() % v.size()] = static_cast<::std::uint8_t>(rng());
      break;
    }
    case 2: { // interesting byte
      constexpr ::std::uint8_t const kInteresting[]{
          0,          1,          0x7F,       0x80,       0xFF,
          'H',        'E',        '.',        '?',        '@',
          '(',        ')',        '[',        ']',        '\\'};
      v[rng() % v.size()] =
          kInteresting[static_cast<::std::size_t>(rng() %
                                                 (sizeof(kInteresting) /
                                                  sizeof(kInteresting[0])))];
      break;
    }
    case 3: { // truncate
      if (v.size() > 1u) {
        v.resize(1u + rng() % v.size());
      }
      break;
    }
    case 4: { // insert chunk of another size (caller keeps seeds)
      break;
    }
    default: { // duplicate a byte somewhere
      v.insert(v.begin() + static_cast<long>(rng() % v.size()),
               v[rng() % v.size()]);
      break;
    }
    }
  }
}

} // namespace herbceptions_fuzz

int main(int argc, char **argv) {
  ::std::vector<::std::vector<::std::uint8_t>> seeds;
  ::std::size_t iterations{0};
  for (int i = 1; i < argc; ++i) {
    if (i == 1) {
      char *end{};
      unsigned long long const parsed{::std::strtoull(argv[i], &end, 10)};
      if (end != argv[i] && *end == '\0') {
        iterations = static_cast<::std::size_t>(parsed);
        continue;
      }
    }
    ::std::FILE *f{::std::fopen(argv[i], "rb")};
    if (f == nullptr) {
      ::std::fprintf(stderr, "cannot open %s\n", argv[i]);
      continue;
    }
    ::std::vector<::std::uint8_t> input;
    char chunk[4096];
    for (::std::size_t got{};
         (got = ::std::fread(chunk, 1, sizeof(chunk), f)) != 0;) {
      input.insert(input.end(), chunk, chunk + got);
    }
    ::std::fclose(f);
    seeds.push_back(::std::move(input));
  }

  if (seeds.empty()) { // replay stdin once
    ::std::vector<::std::uint8_t> input;
    char chunk[4096];
    for (::std::size_t got{};
         (got = ::std::fread(chunk, 1, sizeof(chunk), stdin)) != 0;) {
      input.insert(input.end(), chunk, chunk + got);
    }
    LLVMFuzzerTestOneInput(input.data(), input.size());
    return 0;
  }

  ::std::mt19937_64 rng{0x20260823ull};
  for (::std::size_t i = 0; i != iterations; ++i) {
    ::std::vector<::std::uint8_t> input{
        seeds[rng() % seeds.size()]};
    herbceptions_fuzz::mutate(input, rng);
    LLVMFuzzerTestOneInput(input.data(), input.size());
    if ((i + 1u) % 10000u == 0u) {
      ::std::fprintf(stderr, "iter %zu\n", i + 1u);
    }
  }
  ::std::fprintf(stderr, "done %zu iterations\n", iterations);
  return 0;
}

#endif
