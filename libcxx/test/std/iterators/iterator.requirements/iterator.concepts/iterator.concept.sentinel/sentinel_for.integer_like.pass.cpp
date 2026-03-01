//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include <iterator>
#include <concepts>
#include <cstddef>

static_assert(!std::sentinel_for<int, int*>);
static_assert(!std::sentinel_for<long, int*>);
static_assert(!std::sentinel_for<std::ptrdiff_t, int*>);
static_assert(!std::sentinel_for<unsigned, int*>);

// sanity checks

static_assert(std::sentinel_for<int*, int*>);
static_assert(std::sentinel_for<const int*, int*>);

int main() {}
