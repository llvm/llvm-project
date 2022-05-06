//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcxx-no-debug-mode, c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1

// test container debugging

#include <forward_list>
#include <list>
#include <vector>
#include <deque>
#include "check_assertion.h"
#include "container_debug_tests.h"
#include "test_macros.h"

using namespace IteratorDebugChecks;

template <class Container, ContainerType CT>
struct SequenceContainerChecks : BasicContainerChecks<Container, CT> {
  using Base = BasicContainerChecks<Container, CT>;
  using value_type = typename Container::value_type;
  using allocator_type = typename Container::allocator_type;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;

  using Base::makeContainer;
  using Base::makeValueType;
public:
  static void run() {
    Base::run();
    SanityTest();
    FrontOnEmptyContainer();

    if constexpr(CT != CT_ForwardList) {
        AssignInvalidates();
        BackOnEmptyContainer();
        InsertIterValue();
        InsertIterSizeValue();
        InsertIterIterIter();
        EmplaceIterValue();
        EraseIterIter();
      }
    else {
      SpliceFirstElemAfter();
    }
    if constexpr (CT == CT_Vector || CT == CT_Deque || CT == CT_List) {
      PopBack();
    }
    if constexpr (CT == CT_List || CT == CT_Deque) {
      PopFront(); // FIXME: Run with forward list as well
    }
    if constexpr (CT == CT_List || CT == CT_ForwardList) {
      RemoveFirstElem();
    }
    if constexpr (CT == CT_List) {
      SpliceFirstElem();
      SpliceSameContainer();
    }
  }

private:
  static void SanityTest() {
    // sanity test
    Container C = {1, 1, 1, 1};
    ::DoNotOptimize(&C);
  }

  static void RemoveFirstElem() {
    // See llvm.org/PR35564
    // remove(<first-elem>)
    {
      Container C = makeContainer(1);
      auto FirstVal = *(C.begin());
      C.remove(FirstVal);
      assert(C.empty());
    }
    {
      Container C = {1, 1, 1, 1};
      auto FirstVal = *(C.begin());
      C.remove(FirstVal);
      assert(C.empty());
    }
  }

  static void SpliceFirstElem() {
    // See llvm.org/PR35564
    // splice(<first-elem>)
    {
      Container C = makeContainer(1);
      Container C2;
      C2.splice(C2.end(), C, C.begin(), ++C.begin());
    }
    {
      Container C = makeContainer(1);
      Container C2;
      C2.splice(C2.end(), C, C.begin());
    }
  }

  static void SpliceSameContainer() {
    // splice(<same-container>)
    Container C = {1, 1};
    C.splice(C.end(), C, C.begin());
  }

  static void SpliceFirstElemAfter() {
    // See llvm.org/PR35564
    // splice(<first-elem>)
    {
      Container C = makeContainer(1);
      Container C2;
      C2.splice_after(C2.begin(), C, C.begin(), ++C.begin());
    }
    {
      Container C = makeContainer(1);
      Container C2;
      C2.splice_after(C2.begin(), C, C.begin());
    }
  }

  static void AssignInvalidates() {
    // assign(Size, Value)
    Container C(allocator_type{});
    iterator it1, it2, it3;
    auto reset = [&]() {
      C = makeContainer(3);
      it1 = C.begin();
      it2 = ++C.begin();
      it3 = C.end();
    };
    auto check = [&]() {
      EXPECT_DEATH( C.erase(it1) );
      EXPECT_DEATH( C.erase(it2) );
      EXPECT_DEATH( C.erase(it3, C.end()) );
    };
    reset();
    C.assign(2, makeValueType(4));
    check();
    reset();
    // assign(Iter, Iter)
    std::vector<value_type> V = {
        makeValueType(1),
        makeValueType(2),
        makeValueType(3)
    };
    C.assign(V.begin(), V.end());
    check();
    reset();
    // assign(initializer_list)
    C.assign({makeValueType(1), makeValueType(2), makeValueType(3)});
    check();
  }

  static void BackOnEmptyContainer() {
    // testing back on empty
    Container C = makeContainer(1);
    Container const& CC = C;
    (void)C.back();
    (void)CC.back();
    C.clear();
    EXPECT_DEATH( C.back() );
    EXPECT_DEATH( CC.back() );
  }

  static void FrontOnEmptyContainer() {
    // testing front on empty
    Container C = makeContainer(1);
    Container const& CC = C;
    (void)C.front();
    (void)CC.front();
    C.clear();
    EXPECT_DEATH( C.front() );
    EXPECT_DEATH( CC.front() );
  }

  static void EraseIterIter() {
    // testing erase iter iter invalidation
    Container C1 = makeContainer(3);
    iterator it1 = C1.begin();
    iterator it1_next = ++C1.begin();
    iterator it1_after_next = ++C1.begin();
    ++it1_after_next;
    iterator it1_back = --C1.end();
    assert(it1_next != it1_back);
    if (CT == CT_Vector) {
      EXPECT_DEATH( C1.erase(it1_next, it1) ); // bad range
    }
    C1.erase(it1, it1_after_next);
    EXPECT_DEATH( C1.erase(it1) );
    EXPECT_DEATH( C1.erase(it1_next) );
    if (CT == CT_List) {
      C1.erase(it1_back);
    } else {
      EXPECT_DEATH( C1.erase(it1_back) );
    }
  }

  static void PopBack() {
    // testing  pop_back() invalidation
    Container C1 = makeContainer(2);
    iterator it1 = C1.end();
    --it1;
    C1.pop_back();
    EXPECT_DEATH( C1.erase(it1) );
    C1.erase(C1.begin());
    assert(C1.size() == 0);
    EXPECT_DEATH( C1.pop_back() );
  }

  static void PopFront() {
    // testing pop_front() invalidation
    Container C1 = makeContainer(2);
    iterator it1 = C1.begin();
    C1.pop_front();
    EXPECT_DEATH( C1.erase(it1) );
    C1.erase(C1.begin());
    assert(C1.size() == 0);
    EXPECT_DEATH( C1.pop_front() );
  }

  static void InsertIterValue() {
    // testing insert(iter, value)
    Container C1 = makeContainer(2);
    iterator it1 = C1.begin();
    iterator it1_next = it1;
    ++it1_next;
    Container C2 = C1;
    const value_type value = makeValueType(3);
    value_type rvalue = makeValueType(3);
    EXPECT_DEATH( C2.insert(it1, value) ); // wrong container
    EXPECT_DEATH( C2.insert(it1, std::move(rvalue)) ); // wrong container
    C1.insert(it1_next, value);
    if  (CT == CT_List) {
      C1.insert(it1_next, value);
      C1.insert(it1, value);
      C1.insert(it1_next, std::move(rvalue));
      C1.insert(it1, std::move(rvalue));
    } else {
      EXPECT_DEATH( C1.insert(it1_next, value) ); // invalidated iterator
      EXPECT_DEATH( C1.insert(it1, value) ); // invalidated iterator
      EXPECT_DEATH( C1.insert(it1_next, std::move(rvalue)) ); // invalidated iterator
      EXPECT_DEATH( C1.insert(it1, std::move(rvalue)) ); // invalidated iterator
    }
  }

  static void EmplaceIterValue() {
    // testing emplace(iter, value)
    Container C1 = makeContainer(2);
    iterator it1 = C1.begin();
    iterator it1_next = it1;
    ++it1_next;
    Container C2 = C1;
    const value_type value = makeValueType(3);
    EXPECT_DEATH( C2.emplace(it1, value) ); // wrong container
    EXPECT_DEATH( C2.emplace(it1, makeValueType(4)) ); // wrong container
    C1.emplace(it1_next, value);
    if  (CT == CT_List) {
      C1.emplace(it1_next, value);
      C1.emplace(it1, value);
    } else {
      EXPECT_DEATH( C1.emplace(it1_next, value) ); // invalidated iterator
      EXPECT_DEATH( C1.emplace(it1, value) ); // invalidated iterator
    }
  }

  static void InsertIterSizeValue() {
    // testing insert(iter, size, value)
    Container C1 = makeContainer(2);
    iterator it1 = C1.begin();
    iterator it1_next = it1;
    ++it1_next;
    Container C2 = C1;
    const value_type value = makeValueType(3);
    EXPECT_DEATH( C2.insert(it1, 1, value) ); // wrong container
    C1.insert(it1_next, 2, value);
    if  (CT == CT_List) {
      C1.insert(it1_next, 3, value);
      C1.insert(it1, 1, value);
    } else {
      EXPECT_DEATH( C1.insert(it1_next, 1, value) ); // invalidated iterator
      EXPECT_DEATH( C1.insert(it1, 1, value) ); // invalidated iterator
    }
  }

  static void InsertIterIterIter() {
    // testing insert(iter, iter, iter)
    Container C1 = makeContainer(2);
    iterator it1 = C1.begin();
    iterator it1_next = it1;
    ++it1_next;
    Container C2 = C1;
    std::vector<value_type> V = {
        makeValueType(1),
        makeValueType(2),
        makeValueType(3)
    };
    EXPECT_DEATH( C2.insert(it1, V.begin(), V.end()) ); // wrong container
    C1.insert(it1_next, V.begin(), V.end());
    if  (CT == CT_List) {
      C1.insert(it1_next, V.begin(), V.end());
      C1.insert(it1, V.begin(), V.end());
    } else {
      EXPECT_DEATH( C1.insert(it1_next, V.begin(), V.end()) ); // invalidated iterator
      EXPECT_DEATH( C1.insert(it1, V.begin(), V.end()) ); // invalidated iterator
    }
  }
};

int main(int, char**)
{
  using Alloc = test_allocator<int>;
  {
    SequenceContainerChecks<std::list<int, Alloc>, CT_List>::run();
    SequenceContainerChecks<std::vector<int, Alloc>, CT_Vector>::run();
  }
  // FIXME these containers don't support iterator debugging
  if ((false)) {
    SequenceContainerChecks<
        std::vector<bool, test_allocator<bool>>, CT_VectorBool>::run();
    SequenceContainerChecks<
        std::forward_list<int, Alloc>, CT_ForwardList>::run();
    SequenceContainerChecks<
        std::deque<int, Alloc>, CT_Deque>::run();
  }

  return 0;
}
