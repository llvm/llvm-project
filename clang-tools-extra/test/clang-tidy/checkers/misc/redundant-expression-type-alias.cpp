// RUN: clang-tidy %s -checks=-*,misc-redundant-expression -- -std=c++20 | count 0

#include <array>
#include <tuple>

using MonthArray = std::array<int, 12>;
using ZodiacArray = std::array<int, 12>;

static_assert(std::tuple_size<MonthArray>::value ==
              std::tuple_size<ZodiacArray>::value);
static_assert(std::tuple_size_v<MonthArray> == std::tuple_size_v<ZodiacArray>);

