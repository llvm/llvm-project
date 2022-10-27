//===-- Implementation of getopt ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getopt.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/file.h"
#include "src/__support/common.h"
#include "src/stdio/fprintf.h"

#include <stdio.h>

// This is POSIX compliant and does not support GNU extensions, mainly this is
// just the re-ordering of argv elements such that unknown arguments can be
// easily iterated over.

namespace __llvm_libc {

template <typename T> struct RefWrapper {
  RefWrapper(T *ptr) : ptr(ptr) {}
  RefWrapper &operator=(const RefWrapper &) = default;
  operator T &() { return *ptr; }
  T &get() { return *ptr; }
  T *ptr;
};

struct GetoptContext {
  RefWrapper<char *> optarg;
  RefWrapper<int> optind;
  RefWrapper<int> optopt;
  RefWrapper<unsigned> optpos;

  int opterr;

  FILE *errstream;

  GetoptContext &operator=(const GetoptContext &) = default;

  template <typename... Ts> void report_error(const char *fmt, Ts... ts) {
    if (opterr)
      __llvm_libc::fprintf(errstream, fmt, ts...);
  }
};

struct OptstringParser {
  using value_type = struct {
    char c;
    bool arg;
  };

  cpp::string_view optstring;

  struct iterator {
    cpp::string_view curr;

    iterator operator++() {
      curr = curr.substr(1);
      return *this;
    }

    bool operator!=(iterator other) { return curr.data() != other.curr.data(); }

    value_type operator*() {
      value_type r{curr.front(), false};
      if (!curr.substr(1).empty() && curr.substr(1).front() == ':') {
        this->operator++();
        r.arg = true;
      }
      return r;
    }
  };

  iterator begin() {
    bool skip = optstring.front() == '-' || optstring.front() == '+' ||
                optstring.front() == ':';
    return {optstring.substr(!!skip)};
  }

  iterator end() { return {optstring.substr(optstring.size())}; }
};

int getopt_r(int argc, char *const argv[], const char *optstring,
             GetoptContext &ctx) {
  auto failure = [&ctx](int ret = -1) {
    ctx.optpos.get() = 0;
    return ret;
  };

  if (ctx.optind >= argc || !argv[ctx.optind])
    return failure();

  cpp::string_view current =
      cpp::string_view{argv[ctx.optind]}.substr(ctx.optpos);

  auto move_forward = [&current, &ctx] {
    current = current.substr(1);
    ctx.optpos.get()++;
  };

  // If optpos is nonzero, then we are already parsing a valid flag and these
  // need not be checked.
  if (ctx.optpos == 0) {
    if (current[0] != '-')
      return failure();

    if (current == "--") {
      ctx.optind.get()++;
      return failure();
    }

    // Eat the '-' char.
    move_forward();
    if (current.empty())
      return failure();
  }

  auto find_match =
      [current, optstring]() -> cpp::optional<OptstringParser::value_type> {
    for (auto i : OptstringParser{optstring})
      if (i.c == current[0])
        return i;
    return {};
  };

  auto match = find_match();
  if (!match) {
    ctx.report_error("%s: illegal option -- %c\n", argv[0], current[0]);
    ctx.optopt.get() = current[0];
    return failure('?');
  }

  // We've matched so eat that character.
  move_forward();
  if (match->arg) {
    // If we found an option that takes an argument and our current is not over,
    // the rest of current is that argument. Ie, "-cabc" with opstring "c:",
    // then optarg should point to "abc". Otherwise the argument to c will be in
    // the next arg like "-c abc".
    if (!current.empty()) {
      // This const cast is fine because current was already holding a mutable
      // string, it just doesn't have the semantics to note that, we could use
      // span but it doesn't have string_view string niceties.
      ctx.optarg.get() = const_cast<char *>(current.data());
    } else {
      // One char lookahead to see if we ran out of arguments. If so, return ':'
      // if the first character of optstring is ':'. optind must stay at the
      // current value so only increase it after we known there is another arg.
      if (ctx.optind + 1 >= argc || !argv[ctx.optind + 1]) {
        ctx.report_error("%s: option requires an argument -- %c\n", argv[0],
                         match->c);
        return failure(optstring[0] == ':' ? ':' : '?');
      }
      ctx.optarg.get() = argv[++ctx.optind];
    }
    ctx.optind++;
    ctx.optpos.get() = 0;
  } else if (current.empty()) {
    // If this argument is now empty we are safe to move onto the next one.
    ctx.optind++;
    ctx.optpos.get() = 0;
  }

  return match->c;
}

namespace impl {

extern "C" char *optarg;
extern "C" int optind = 1;
extern "C" int optopt;
extern "C" int opterr;

static unsigned optpos;

static GetoptContext ctx{
    &impl::optarg, &impl::optind,
    &impl::optopt, &optpos,
    impl::opterr,  reinterpret_cast<FILE *>(__llvm_libc::stderr)};

#ifndef LLVM_LIBC_PUBLIC_PACKAGING
// This is used exclusively in tests.
void set_getopt_state(char **optarg, int *optind, int *optopt, unsigned *optpos,
                      int opterr, FILE *errstream) {
  ctx = {optarg, optind, optopt, optpos, opterr, errstream};
}
#endif

} // namespace impl

LLVM_LIBC_FUNCTION(int, getopt,
                   (int argc, char *const argv[], const char *optstring)) {
  return getopt_r(argc, argv, optstring, impl::ctx);
}

} // namespace __llvm_libc
