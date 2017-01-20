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

//=============================================================================
// TESTING std::unique_ptr::unique_ptr()
//
// Concerns:
//   1 The default constructor works for any default constructible deleter types.
//   2 The stored type 'T' is allowed to be incomplete.
//
// Plan
//  1 Default construct unique_ptr's with various deleter types (C-1)
//  2 Default construct a unique_ptr with an incomplete element_type and
//    various deleter types (C-1,2)

#include <memory>
#include <cassert>

#include "deleter_types.h"

struct IncompleteT;

void checkNumIncompleteTypeAlive(int i);

template <class Del = std::default_delete<IncompleteT> >
struct StoresIncomplete {
  std::unique_ptr<IncompleteT, Del> m_ptr;
  StoresIncomplete() {}
  ~StoresIncomplete();

  IncompleteT* get() const { return m_ptr.get(); }
  Del& get_deleter() { return m_ptr.get_deleter(); }
};

int main()
{
    {
      std::unique_ptr<int> p;
      assert(p.get() == 0);
    }
    {
      std::unique_ptr<int, NCDeleter<int> > p;
      assert(p.get() == 0);
      assert(p.get_deleter().state() == 0);
      p.get_deleter().set_state(5);
      assert(p.get_deleter().state() == 5);
    }
    {
        StoresIncomplete<> s;
        assert(s.get() == 0);
        checkNumIncompleteTypeAlive(0);
    }
    checkNumIncompleteTypeAlive(0);
    {
        StoresIncomplete< Deleter<IncompleteT> > s;
        assert(s.get() == 0);
        assert(s.get_deleter().state() == 0);
        checkNumIncompleteTypeAlive(0);
    }
    checkNumIncompleteTypeAlive(0);
}

struct IncompleteT {
    static int count;
    IncompleteT() { ++count; }
    ~IncompleteT() {--count; }
};

int IncompleteT::count = 0;

void checkNumIncompleteTypeAlive(int i) {
    assert(IncompleteT::count == i);
}

template <class Del>
StoresIncomplete<Del>::~StoresIncomplete() { }
