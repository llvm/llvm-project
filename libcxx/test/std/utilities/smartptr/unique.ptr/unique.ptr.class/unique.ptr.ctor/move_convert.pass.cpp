//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <memory>

// unique_ptr

// Test unique_ptr converting move ctor

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <int ID = 0>
struct GenericDeleter {
  void operator()(void*) const {}
};

template <int ID = 0>
struct GenericConvertingDeleter {
  template <int OID>
  GenericConvertingDeleter(GenericConvertingDeleter<OID>) {}
  void operator()(void*) const {}
};

template <bool IsArray>
void test_sfinae() {
#if TEST_STD_VER >= 11
  typedef typename std::conditional<IsArray, A[], A>::type VT;

  { // Test that different non-reference deleter types are allowed so long
    // as they convert to each other.
    using U1 = std::unique_ptr<VT, GenericConvertingDeleter<0> >;
    using U2 = std::unique_ptr<VT, GenericConvertingDeleter<1> >;
    static_assert(std::is_constructible<U1, U2&&>::value, "");
  }
  { // Test that different non-reference deleter types are disallowed when
    // they cannot convert.
    using U1 = std::unique_ptr<VT, GenericDeleter<0> >;
    using U2 = std::unique_ptr<VT, GenericDeleter<1> >;
    static_assert(!std::is_constructible<U1, U2&&>::value, "");
  }
  { // Test that if the destination deleter is a reference type then only
    // exact matches are allowed.
    using U1 = std::unique_ptr<VT, GenericConvertingDeleter<0> const& >;
    using U2 = std::unique_ptr<VT, GenericConvertingDeleter<0> >;
    using U3 = std::unique_ptr<VT, GenericConvertingDeleter<0> &>;
    using U4 = std::unique_ptr<VT, GenericConvertingDeleter<1> >;
    using U5 = std::unique_ptr<VT, GenericConvertingDeleter<1> const&>;
    static_assert(!std::is_constructible<U1, U2&&>::value, "");
    static_assert(!std::is_constructible<U1, U3&&>::value, "");
    static_assert(!std::is_constructible<U1, U4&&>::value, "");
    static_assert(!std::is_constructible<U1, U5&&>::value, "");

    using U1C = std::unique_ptr<const VT, GenericConvertingDeleter<0> const&>;
    static_assert(std::is_nothrow_constructible<U1C, U1&&>::value, "");
  }
  { // Test that non-reference destination deleters can be constructed
    // from any source deleter type with a sutible conversion. Including
    // reference types.
    using U1 = std::unique_ptr<VT, GenericConvertingDeleter<0> >;
    using U2 = std::unique_ptr<VT, GenericConvertingDeleter<0> &>;
    using U3 = std::unique_ptr<VT, GenericConvertingDeleter<0> const &>;
    using U4 = std::unique_ptr<VT, GenericConvertingDeleter<1> >;
    using U5 = std::unique_ptr<VT, GenericConvertingDeleter<1> &>;
    using U6 = std::unique_ptr<VT, GenericConvertingDeleter<1> const&>;
    static_assert(std::is_constructible<U1, U2&&>::value, "");
    static_assert(std::is_constructible<U1, U3&&>::value, "");
    static_assert(std::is_constructible<U1, U4&&>::value, "");
    static_assert(std::is_constructible<U1, U5&&>::value, "");
    static_assert(std::is_constructible<U1, U6&&>::value, "");
  }
#endif
}


template <bool IsArray>
void test_noexcept() {
#if TEST_STD_VER >= 11
  typedef typename std::conditional<IsArray, A[], A>::type VT;
  {
    typedef std::unique_ptr<const VT> APtr;
    typedef std::unique_ptr<VT> BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef std::unique_ptr<const VT, CDeleter<const VT> > APtr;
    typedef std::unique_ptr<VT, CDeleter<VT> > BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef std::unique_ptr<const VT, NCDeleter<const VT>&> APtr;
    typedef std::unique_ptr<VT, NCDeleter<const VT>&> BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef std::unique_ptr<const VT, const NCConstDeleter<const VT>&> APtr;
    typedef std::unique_ptr<VT, const NCConstDeleter<const VT>&> BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
#endif
}


int main() {
  {
    test_sfinae</*IsArray*/false>();
    test_noexcept<false>();
  }
  {
    test_sfinae</*IsArray*/true>();
    test_noexcept<true>();
  }
}
