//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Check that functions are marked [[nodiscard]]

#include <cstddef>
#include <ranges>
#include <utility>
#include <vector>

// begin()

struct NonMemberBegin {};
int* begin(NonMemberBegin);

// end()

struct NonMemberBeginEnd {};
int* begin(NonMemberBeginEnd);
int* end(NonMemberBeginEnd);
template <>
inline constexpr bool std::ranges::enable_borrowed_range<NonMemberBeginEnd> = true;

// cbegin()

struct MemberConstBegin {
  int* begin() const;
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<MemberConstBegin> = true;

// cend()

struct MemberConstBeginConstEnd {
  int* begin() const;
  int* end() const;
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<MemberConstBeginConstEnd> = true;

// rbegin()

struct MemberRBegin {
  int* rbegin();
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<MemberRBegin> = true;

struct NonMemberRBegin {};
int* rbegin(NonMemberRBegin);
template <>
inline constexpr bool std::ranges::enable_borrowed_range<NonMemberRBegin> = true;

// crbegin()

struct MemberConstRBegin {
  int* rbegin() const;
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<MemberConstRBegin> = true;

// rend()

struct MemberREnd {
  int* rbegin();
  int* rend();
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<MemberREnd> = true;

struct NonMemberREnd {};
int* rbegin(NonMemberREnd);
int* rend(NonMemberREnd);
template <>
inline constexpr bool std::ranges::enable_borrowed_range<NonMemberREnd> = true;

// crend()

struct MemberConstREnd {
  int* rbegin() const;
  int* rend() const;
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<MemberConstREnd> = true;

// size()

struct NonMemberSize {};
std::size_t size(NonMemberSize) { return 0; }

struct DisableSizedRange {
  int* begin();
  int* end();
  std::size_t size() { return 0; }
};

template <>
inline constexpr bool std::ranges::disable_sized_range<DisableSizedRange> = true;

// data()

struct BorrowedRange {
  int* begin() const { return nullptr; }
};
template <>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;

void test() {
  // [range.access.begin]

  {
    extern int uArr[];
    int arr[]{94, 82, 47};

    struct MemberBegin {
      int* begin() { return nullptr; };
    } member;

    NonMemberBegin nonMember;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::begin(uArr);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::begin(arr);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::begin(member);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::begin(nonMember);
  }

  // [range.access.end]

  {
    int arr[]{94, 82, 47};

    struct MemberBeginEnd {
      int* begin() { return nullptr; };
      int* end() { return nullptr; };
    } member;

    NonMemberBeginEnd nonMember;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::end(arr);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::end(member);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::end(nonMember);
  }

  // [range.access.cbegin]

  {
    MemberConstBegin obj;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::cbegin(std::as_const(obj));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::cbegin(std::move(obj));
  }

  // [range.access.cend]

  {
    MemberConstBeginConstEnd obj;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::cend(std::as_const(obj));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::cend(std::move(obj));
  }

  // [range.access.rbegin]

  {
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::rbegin(MemberRBegin{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::rbegin(NonMemberRBegin{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::rbegin(NonMemberBeginEnd{});
  }

  // [range.access.rend]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::rend(MemberREnd{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::rend(NonMemberREnd{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::ranges::rend(NonMemberBeginEnd{});

  // [range.access.crbegin]

  {
    MemberConstRBegin obj;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::crbegin(std::as_const(obj));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::crbegin(std::move(obj));
  }

  // [range.access.crend]
  {
    MemberConstREnd obj;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::crend(std::as_const(obj));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::crend(std::move(obj));
  }

  // [range.prim.size]
  {
    int arr[]{94, 82, 49};

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::size(std::move(arr));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::size(arr);

    struct MemberSize {
      std::size_t size() const { return 0; };
    };

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::size(MemberSize{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::size(NonMemberSize{});

    DisableSizedRange ds;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::size(ds);
  }

  // [range.prim.ssize]
  {
    struct UnsignedReturnTypeSize {
      constexpr unsigned short size() { return 0; }
    };

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::ssize(UnsignedReturnTypeSize{});
  }

  // [range.prim.size.hint]

  // [range.prim.empty]

  {
    struct MemberEmptyRange {
      int* begin();
      bool empty() const { return true; };
    };

    struct MemberSizeRange {
      int* begin() { return nullptr; };
      std::size_t size() const { return 0; }
    };

    struct BeginEndComparableRange {
      struct Sentinel {
        constexpr bool operator==(int*) const { return true; }
      };

      int* begin() { return nullptr; };
      Sentinel end() { return {}; };
    };

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::empty(MemberEmptyRange{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::empty(MemberSizeRange{});

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::empty(BeginEndComparableRange{});
  }

  // [range.prim.data]

  {
    struct MemberDataRange {
      int* begin();
      int* data() { return nullptr; }
    } memberDataRange;

    struct MemberBeginRange {
      int* begin() { return nullptr; }
    } range;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::data(memberDataRange);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::data(range);
  }

  // [range.prim.cdata]

  {
    struct MemberConstBeginRange {
      int* begin() const { return nullptr; }
    } range;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::cdata(range);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::cdata(BorrowedRange{});
  }
}
