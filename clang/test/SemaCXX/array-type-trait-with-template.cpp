// RUN: %clang_cc1 -fsyntax-only %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -DWITH_AUTO_FUNCTION_PARAMETER=1 %s

// When __array_rank is used with a template type parameter, this test
// ensures clang considers the final expression could be used with
// static_assert/constexpr.
//
// Although array_extent was handled well, we add it as a precaution.

template <typename T>
using remove_reference_t = __remove_reference_t(T);

template <typename T, int N>
constexpr int array_rank(T (&lhs)[N]) {
  return __array_rank(T[N]);
}

template <int I, typename T, int N>
 constexpr int array_extent(T (&lhs)[N]) {
  return __array_extent(T[N], I);
}

template <typename T>
struct Rank {
  using ArrayT = remove_reference_t<T>;

  template <int N>
  static constexpr int call(ArrayT (&lhs)[N]) {
    return __array_rank(ArrayT[N]);
  }
};

template <typename T>
struct Extent {
  using ArrayT = remove_reference_t<T>;

  template <int I, int N>
  static constexpr int call(ArrayT (&lhs)[N]) {
    return __array_extent(ArrayT[N], I);
  }
};

#ifdef WITH_AUTO_FUNCTION_PARAMETER
template <int N>
constexpr int array_rank_auto(auto (&lhs)[N]) {
  return __array_rank(remove_reference_t<decltype(lhs[0])>[N]);
}

template <int I, int N>
constexpr int array_extent_auto(auto (&lhs)[N]) {
  return __array_extent(remove_reference_t<decltype(lhs[0])>[N], I);
}
#endif

template <int N>
constexpr int array_rank_int(const int (&lhs)[N]) {
  return __array_rank(const int[N]);
}

template <int I, int N>
constexpr int array_extent_int(const int (&lhs)[N]) {
  return __array_extent(const int[N], I);
}

template <int M, int N>
constexpr int array_rank_int(const int (&lhs)[M][N]) {
  return __array_rank(const int[M][N]);
}

template <int I, int M, int N>
constexpr int array_extent_int(const int (&lhs)[M][N]) {
  return __array_extent(const int[M][N], I);
}

int main() {
  constexpr int vec[] = {0, 1, 2, 1};
  constexpr int mat[4][4] = {
    {1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
  };

#define ATT_TESTS_WITH_ASSERT(ATT_ASSERT)	\
  { ATT_ASSERT(RANK(vec) == 1);	}		\
  { ATT_ASSERT(RANK(mat) == 2);	}		\
  { ATT_ASSERT(EXTENT(vec, 0) == 4); }		\
  { ATT_ASSERT(EXTENT(vec, 1) == 0); }		\
  { ATT_ASSERT(EXTENT(mat, 1) == 4); }

#define ATT_TESTS()				\
  ATT_TESTS_WITH_ASSERT( constexpr bool cst = )	\
  ATT_TESTS_WITH_ASSERT( (void) )		\
  ATT_TESTS_WITH_ASSERT( static_assert )

  {
#define RANK(lhs) array_rank(lhs)
#define EXTENT(lhs, i) array_extent<i>(lhs)
    ATT_TESTS();
#undef RANK
#undef EXTENT
  }

  {
#define RANK(lhs) Rank<decltype(lhs[0])>::call(lhs)
#define EXTENT(lhs, i) Extent<decltype(lhs[0])>::call<i>(lhs)
    ATT_TESTS();
#undef RANK
#undef EXTENT
  }

#ifdef WITH_AUTO_FUNCTION_PARAMETER
  {
#define RANK(lhs) array_rank_auto(lhs)
#define EXTENT(lhs, i) array_extent_auto<i>(lhs)
    ATT_TESTS();
#undef RANK
#undef EXTENT
  }
#endif

  {
#define RANK(lhs) array_rank_int(lhs)
#define EXTENT(lhs, i) array_extent_int<i>(lhs)
    ATT_TESTS();
#undef RANK
#undef EXTENT
  }
}
