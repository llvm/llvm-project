//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::data
// std::ranges::cdata

#include <ranges>

#include <cassert>
#include <type_traits>
#include "test_macros.h"
#include "test_iterators.h"

using RangeDataT = decltype(std::ranges::data);
using RangeCDataT = decltype(std::ranges::cdata);

#if TEST_STD_VER < 23
#  define STD_VER_20(...) __VA_ARGS__
#  define STD_VER_23(...) static_assert(true)
#else
#  define STD_VER_20(...) static_assert(true)
#  define STD_VER_23(...) __VA_ARGS__
#endif

static int globalBuff[2];

struct Incomplete;

static_assert(!std::is_invocable_v<RangeDataT, Incomplete[]>);
static_assert(!std::is_invocable_v<RangeDataT, Incomplete(&&)[2]>);
static_assert(!std::is_invocable_v<RangeDataT, Incomplete(&&)[2][2]>);
static_assert(!std::is_invocable_v<RangeDataT, int [1]>);
static_assert(!std::is_invocable_v<RangeDataT, int (&&)[1]>);
static_assert( std::is_invocable_v<RangeDataT, int (&)[1]>);

static_assert(!std::is_invocable_v<RangeCDataT, Incomplete[]>);
static_assert(!std::is_invocable_v<RangeCDataT, Incomplete(&&)[2]>);
static_assert(!std::is_invocable_v<RangeCDataT, Incomplete(&&)[2][2]>);
static_assert(!std::is_invocable_v<RangeCDataT, int [1]>);
static_assert(!std::is_invocable_v<RangeCDataT, int (&&)[1]>);
static_assert( std::is_invocable_v<RangeCDataT, int (&)[1]>);

struct DataMemberNonRange {
  int x;
  constexpr const int* data() const { return &x; }
};
static_assert(std::is_invocable_v<RangeDataT, DataMemberNonRange&>);
static_assert(!std::is_invocable_v<RangeDataT, DataMemberNonRange&&>);
static_assert(std::is_invocable_v<RangeDataT, DataMemberNonRange const&>);
static_assert(!std::is_invocable_v<RangeDataT, DataMemberNonRange const&&>);
#if TEST_STD_VER < 23
static_assert(std::is_invocable_v<RangeCDataT, DataMemberNonRange&>);
static_assert(!std::is_invocable_v<RangeCDataT, DataMemberNonRange&&>);
static_assert(std::is_invocable_v<RangeCDataT, DataMemberNonRange const&>);
static_assert(!std::is_invocable_v<RangeCDataT, DataMemberNonRange const&&>);
#endif

struct DataMember {
  int x;
  constexpr const int *data() const { return &x; }
  const int* begin() const;
  const int* end() const;
};
static_assert( std::is_invocable_v<RangeDataT, DataMember &>);
static_assert(!std::is_invocable_v<RangeDataT, DataMember &&>);
static_assert( std::is_invocable_v<RangeDataT, DataMember const&>);
static_assert(!std::is_invocable_v<RangeDataT, DataMember const&&>);
static_assert( std::is_invocable_v<RangeCDataT, DataMember &>);
static_assert(!std::is_invocable_v<RangeCDataT, DataMember &&>);
static_assert( std::is_invocable_v<RangeCDataT, DataMember const&>);
static_assert(!std::is_invocable_v<RangeCDataT, DataMember const&&>);

constexpr bool testReturnTypes() {
  {
    int *x[2];
    ASSERT_SAME_TYPE(decltype(std::ranges::data(x)), int**);
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(x)), int* const*);
  }
  {
    int x[2][2];
    ASSERT_SAME_TYPE(decltype(std::ranges::data(x)), int(*)[2]);
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(x)), const int(*)[2]);
  }
  {
    struct D {
      char*& data();
      short*& data() const;
      int* begin();
      int* end();
    };
    ASSERT_SAME_TYPE(decltype(std::ranges::data(std::declval<D&>())), char*);
    static_assert(!std::is_invocable_v<RangeDataT, D&&>);
    ASSERT_SAME_TYPE(decltype(std::ranges::data(std::declval<const D&>())), short*);
    static_assert(!std::is_invocable_v<RangeDataT, const D&&>);

    static_assert(!std::is_invocable_v<RangeCDataT, D&&>);
    static_assert(!std::is_invocable_v<RangeCDataT, const D&&>);
    STD_VER_20(ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<D&>())), short*));
    STD_VER_23(ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<D&>())), const char*));
    STD_VER_20(ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<const D&>())), short*));
    STD_VER_23(static_assert(!std::is_invocable_v<RangeCDataT, const D&>));
  }
#if TEST_STD_VER >= 23
  {
    struct D {
      char*& data();
      short*& data() const;
      int* begin();
      int* end();
      int* begin() const;
      int* end() const;
    };
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<D&>())), const short*);
    static_assert(!std::is_invocable_v<RangeCDataT, D&&>);
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<const D&>())), const short*);
    static_assert(!std::is_invocable_v<RangeCDataT, const D&&>);
  }
  {
    struct D {
      char*& data();
      short*& data() const;
      int* begin();
      int* end();
      const int* begin() const;
      const int* end() const;
    };
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<D&>())), const short*);
    static_assert(!std::is_invocable_v<RangeCDataT, D&&>);
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<const D&>())), const short*);
    static_assert(!std::is_invocable_v<RangeCDataT, const D&&>);
  }
#endif // TEST_STD_VER >= 23
  {
    struct NC {
      char *begin() const;
      char *end() const;
      int *data();
    };
    static_assert(!std::ranges::contiguous_range<NC>);
    static_assert( std::ranges::contiguous_range<const NC>);
    ASSERT_SAME_TYPE(decltype(std::ranges::data(std::declval<NC&>())), int*);
    static_assert(!std::is_invocable_v<RangeDataT, NC&&>);
    ASSERT_SAME_TYPE(decltype(std::ranges::data(std::declval<const NC&>())), char*);
    static_assert(!std::is_invocable_v<RangeDataT, const NC&&>);

    static_assert(!std::is_invocable_v<RangeCDataT, NC&&>);
    static_assert(!std::is_invocable_v<RangeCDataT, const NC&&>);
#if TEST_STD_VER < 23
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<NC&>())), char*);
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<const NC&>())), char*);
#else
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<NC&>())), const char*);
    ASSERT_SAME_TYPE(decltype(std::ranges::cdata(std::declval<const NC&>())), const char*);
#endif
  }
  return true;
}

struct VoidDataMember {
  void *data() const;
};
static_assert(!std::is_invocable_v<RangeDataT, VoidDataMember const&>);
static_assert(!std::is_invocable_v<RangeCDataT, VoidDataMember const&>);

struct Empty { };
struct EmptyDataMember {
  Empty data() const;
};
static_assert(!std::is_invocable_v<RangeDataT, EmptyDataMember const&>);
static_assert(!std::is_invocable_v<RangeCDataT, EmptyDataMember const&>);

struct PtrConvertibleDataMember {
  struct Ptr {
    operator int*() const;
  };
  Ptr data() const;
};
static_assert(!std::is_invocable_v<RangeDataT, PtrConvertibleDataMember const&>);
static_assert(!std::is_invocable_v<RangeCDataT, PtrConvertibleDataMember const&>);

struct NonConstDataMember {
  int x;
  constexpr int *data() { return &x; }
  int* begin();
  int* end();
};

struct EnabledBorrowingDataMember {
  constexpr int *data() { return &globalBuff[0]; }
  int* begin();
  int* end();
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<EnabledBorrowingDataMember> = true;

struct DataMemberAndBegin {
  int x;
  constexpr const int *data() const { return &x; }
  const int *begin() const;
  const int* end() const;
};

constexpr bool testDataMember() {
  DataMember a;
  assert(std::ranges::data(a) == &a.x);
  assert(std::ranges::cdata(a) == &a.x);

  NonConstDataMember b;
  assert(std::ranges::data(b) == &b.x);
  STD_VER_20(static_assert(!std::is_invocable_v<RangeCDataT, decltype((b))>));
  STD_VER_23(assert(std::ranges::cdata(b) == &b.x));

  EnabledBorrowingDataMember c;
  assert(std::ranges::data(std::move(c)) == &globalBuff[0]);
  STD_VER_20(static_assert(!std::is_invocable_v<RangeCDataT, decltype((c))>));
  STD_VER_23(assert(std::ranges::data(c) == &globalBuff[0]));

  return true;
}

using ContiguousIter = contiguous_iterator<const int*>;

struct BeginMemberContiguousIterator {
  int buff[8];

  constexpr ContiguousIter begin() const { return ContiguousIter(buff); }
  constexpr ContiguousIter end() const { return ContiguousIter(buff); }
};
static_assert( std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator &>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator &&>);
static_assert( std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&&>);
static_assert( std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator &>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator &&>);
static_assert( std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&&>);

struct BeginMemberRandomAccess {
  int buff[8];

  random_access_iterator<const int*> begin() const;
  random_access_iterator<const int*> end() const;
};
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberRandomAccess&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberRandomAccess&&>);
static_assert(!std::is_invocable_v<RangeDataT, const BeginMemberRandomAccess&>);
static_assert(!std::is_invocable_v<RangeDataT, const BeginMemberRandomAccess&&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberRandomAccess&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberRandomAccess&&>);
static_assert(!std::is_invocable_v<RangeCDataT, const BeginMemberRandomAccess&>);
static_assert(!std::is_invocable_v<RangeCDataT, const BeginMemberRandomAccess&&>);

struct BeginFriendContiguousIterator {
  int buff[8];

  friend constexpr ContiguousIter begin(const BeginFriendContiguousIterator &iter) {
    return ContiguousIter(iter.buff);
  }
  friend constexpr ContiguousIter end(const BeginFriendContiguousIterator& iter) { return ContiguousIter(iter.buff); }
};
static_assert( std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator &>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator &&>);
static_assert( std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberContiguousIterator const&&>);
static_assert( std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator &>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator &&>);
static_assert( std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberContiguousIterator const&&>);

struct BeginFriendRandomAccess {
  friend random_access_iterator<const int*> begin(const BeginFriendRandomAccess& iter);
  friend random_access_iterator<const int*> end(const BeginFriendRandomAccess& iter);
};
static_assert(!std::is_invocable_v<RangeDataT, BeginFriendRandomAccess&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginFriendRandomAccess&&>);
static_assert(!std::is_invocable_v<RangeDataT, const BeginFriendRandomAccess&>);
static_assert(!std::is_invocable_v<RangeDataT, const BeginFriendRandomAccess&&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginFriendRandomAccess&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginFriendRandomAccess&&>);
static_assert(!std::is_invocable_v<RangeCDataT, const BeginFriendRandomAccess&>);
static_assert(!std::is_invocable_v<RangeCDataT, const BeginFriendRandomAccess&&>);

struct BeginMemberRvalue {
  int buff[8];

  ContiguousIter begin() &&;
  ContiguousIter end() &&;
};
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberRvalue&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberRvalue&&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberRvalue const&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberRvalue const&&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberRvalue&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberRvalue&&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberRvalue const&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberRvalue const&&>);

struct BeginMemberBorrowingEnabled {
  constexpr contiguous_iterator<int*> begin() { return contiguous_iterator<int*>{&globalBuff[1]}; }
  constexpr contiguous_iterator<int*> end() { return contiguous_iterator<int*>{&globalBuff[2]}; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BeginMemberBorrowingEnabled> = true;
static_assert( std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled &>);
static_assert( std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled &&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled const&>);
static_assert(!std::is_invocable_v<RangeDataT, BeginMemberBorrowingEnabled const&&>);
STD_VER_20(static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled&>));
STD_VER_23(static_assert(std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled&>));
STD_VER_20(static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled&&>));
STD_VER_23(static_assert(std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled&&>));
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled const&>);
static_assert(!std::is_invocable_v<RangeCDataT, BeginMemberBorrowingEnabled const&&>);

struct ConstBeginMemberBorrowingEnabled {
  constexpr contiguous_iterator<int*> begin() const { return contiguous_iterator<int*>{&globalBuff[1]}; }
  constexpr contiguous_iterator<int*> end() const { return contiguous_iterator<int*>{&globalBuff[2]}; }
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<ConstBeginMemberBorrowingEnabled> = true;
static_assert(std::is_invocable_v<RangeDataT, ConstBeginMemberBorrowingEnabled&>);
static_assert(std::is_invocable_v<RangeDataT, ConstBeginMemberBorrowingEnabled&&>);
static_assert(std::is_invocable_v<RangeDataT, ConstBeginMemberBorrowingEnabled const&>);
static_assert(std::is_invocable_v<RangeDataT, ConstBeginMemberBorrowingEnabled const&&>);
static_assert(std::is_invocable_v<RangeCDataT, ConstBeginMemberBorrowingEnabled&>);
static_assert(std::is_invocable_v<RangeCDataT, ConstBeginMemberBorrowingEnabled&&>);
static_assert(std::is_invocable_v<RangeCDataT, ConstBeginMemberBorrowingEnabled const&>);
static_assert(std::is_invocable_v<RangeCDataT, ConstBeginMemberBorrowingEnabled const&&>);

constexpr bool testViaRangesBegin() {
  int arr[2];
  assert(std::ranges::data(arr) == arr + 0);
  assert(std::ranges::cdata(arr) == arr + 0);

  BeginMemberContiguousIterator a;
  assert(std::ranges::data(a) == a.buff);
  assert(std::ranges::cdata(a) == a.buff);

  const BeginFriendContiguousIterator b {};
  assert(std::ranges::data(b) == b.buff);
  assert(std::ranges::cdata(b) == b.buff);

  BeginMemberBorrowingEnabled c;
  assert(std::ranges::data(std::move(c)) == &globalBuff[1]);
  STD_VER_20(static_assert(!std::is_invocable_v<RangeCDataT, decltype(std::move(c))>));
  STD_VER_23(static_assert(std::is_invocable_v<RangeCDataT, decltype(std::move(c))>));

  ConstBeginMemberBorrowingEnabled d;
  assert(std::ranges::data(std::move(d)) == &globalBuff[1]);
  assert(std::ranges::cdata(std::move(d)) == &globalBuff[1]);

  return true;
}

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeDataT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeDataT, Holder<Incomplete>*&>);
static_assert(!std::is_invocable_v<RangeCDataT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCDataT, Holder<Incomplete>*&>);

struct RandomButNotContiguous {
  random_access_iterator<int*> begin() const;
  random_access_iterator<int*> end() const;
};
static_assert(!std::is_invocable_v<RangeDataT, RandomButNotContiguous>);
static_assert(!std::is_invocable_v<RangeDataT, RandomButNotContiguous&>);
static_assert(!std::is_invocable_v<RangeCDataT, RandomButNotContiguous>);
static_assert(!std::is_invocable_v<RangeCDataT, RandomButNotContiguous&>);

int main(int, char**) {
  static_assert(testReturnTypes());

  testDataMember();
  static_assert(testDataMember());

  testViaRangesBegin();
  static_assert(testViaRangesBegin());

  return 0;
}
