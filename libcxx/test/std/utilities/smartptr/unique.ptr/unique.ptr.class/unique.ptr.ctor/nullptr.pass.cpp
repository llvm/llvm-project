//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// unique_ptr(nullptr_t);

#include <memory>
#include <cassert>

#include "test_macros.h"
#include "deleter_types.h"
#include "unique_ptr_test_helper.h"

#include "test_workarounds.h" // For TEST_WORKAROUND_UPCOMING_UNIQUE_PTR_CHANGES

// default unique_ptr ctor should only require default Deleter ctor
class DefaultDeleter {
  int state_;

  DefaultDeleter(DefaultDeleter&);
  DefaultDeleter& operator=(DefaultDeleter&);

public:
  DefaultDeleter() : state_(5) {}

  int state() const { return state_; }

  void operator()(void*) {}
};

#if TEST_STD_VER >= 11
struct NonDefaultDeleter {
  NonDefaultDeleter() = delete;
  void operator()(void*) const {}
};
#endif

template <class VT>
void test_basic() {
  {
    std::unique_ptr<VT> p(nullptr);
    assert(p.get() == 0);
  }
  {
    std::unique_ptr<VT, DefaultDeleter> p(nullptr);
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 5);
  }
#if TEST_STD_VER >= 11
  {
    using U1 = std::unique_ptr<VT>;
    using U2 = std::unique_ptr<VT, Deleter<VT> >;
    static_assert(std::is_nothrow_constructible<U1, decltype(nullptr)>::value,
                  "");
    static_assert(std::is_nothrow_constructible<U2, decltype(nullptr)>::value,
                  "");
  }
#endif
}

template <class VT>
void test_sfinae() {
#if TEST_STD_VER >= 11 && !defined(TEST_WORKAROUND_UPCOMING_UNIQUE_PTR_CHANGES)
  { // the constructor does not participate in overload resultion when
    // the deleter is a pointer type
    using U = std::unique_ptr<VT, void (*)(void*)>;
    static_assert(!std::is_constructible<U, decltype(nullptr)>::value, "");
  }
  { // the constructor does not participate in overload resolution when
    // the deleter is not default constructible
    using Del = CDeleter<VT>;
    using U1 = std::unique_ptr<VT, NonDefaultDeleter>;
    using U2 = std::unique_ptr<VT, Del&>;
    using U3 = std::unique_ptr<VT, Del const&>;
    static_assert(!std::is_constructible<U1, decltype(nullptr)>::value, "");
    static_assert(!std::is_constructible<U2, decltype(nullptr)>::value, "");
    static_assert(!std::is_constructible<U3, decltype(nullptr)>::value, "");
  }
#endif
}

DEFINE_AND_RUN_IS_INCOMPLETE_TEST({
  { doIncompleteTypeTest(0, nullptr); }
  checkNumIncompleteTypeAlive(0);
  {
    doIncompleteTypeTest<IncompleteType, NCDeleter<IncompleteType> >(0,
                                                                     nullptr);
  }
  checkNumIncompleteTypeAlive(0);
  { doIncompleteTypeTest<IncompleteType[]>(0, nullptr); }
  checkNumIncompleteTypeAlive(0);
  {
    doIncompleteTypeTest<IncompleteType[], NCDeleter<IncompleteType[]> >(
        0, nullptr);
  }
  checkNumIncompleteTypeAlive(0);
})

int main() {
  {
    test_basic<int>();
    test_sfinae<int>();
  }
  {
    test_basic<int[]>();
    test_sfinae<int[]>();
  }
}
