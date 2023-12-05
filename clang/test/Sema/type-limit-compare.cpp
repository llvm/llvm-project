// RUN: %clang_cc1 %s -fsyntax-only -Wtautological-type-limit-compare -verify

// expected-no-diagnostics
#if defined(_WIN32)
typedef unsigned long long uint64_t;
#else
typedef unsigned long uint64_t;
#endif

namespace std {
using size_t = decltype(sizeof(0));
} // namespace std

bool func(uint64_t Size) {
  if (sizeof(std::size_t) < sizeof(uint64_t) &&
     Size > (uint64_t)(__SIZE_MAX__))
    return false;
  return true;
}

