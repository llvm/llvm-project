// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config c++-inlining=destructors -analyzer-config c++-container-inlining=false -verify %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config c++-inlining=destructors -analyzer-config c++-container-inlining=true -DINLINE=1 -verify %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config c++-inlining=destructors -analyzer-config c++-container-inlining=false -DTEST_INLINABLE_ALLOCATORS -verify %s
// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-config c++-inlining=destructors -analyzer-config c++-container-inlining=true -DTEST_INLINABLE_ALLOCATORS -DINLINE=1 -verify %s

#ifndef HEADER

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

#define HEADER
#include "containers.cpp"
#undef HEADER

void test() {
  MySet set(0);

  clang_analyzer_eval(set.isEmpty());
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif

  clang_analyzer_eval(set.raw_begin() == set.raw_end());
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif

  clang_analyzer_eval(set.begin().impl == set.end().impl);
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif
}

void testSubclass(MySetSubclass &sub) {
  sub.useIterator(sub.begin());

  MySetSubclass local;
}

void testWrappers(BeginOnlySet &w1, IteratorStructOnlySet &w2,
                  IteratorTypedefOnlySet &w3, IteratorUsingOnlySet &w4) {
  BeginOnlySet local1;
  IteratorStructOnlySet local2;
  IteratorTypedefOnlySet local3;
  IteratorUsingOnlySet local4;

  clang_analyzer_eval(w1.begin().impl.impl == w1.begin().impl.impl);
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif

  clang_analyzer_eval(w2.start().impl == w2.start().impl);
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif

  clang_analyzer_eval(w3.start().impl == w3.start().impl);
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif

  clang_analyzer_eval(w4.start().impl == w4.start().impl);
#if INLINE
  // expected-warning@-2 {{TRUE}}
#else
  // expected-warning@-4 {{UNKNOWN}}
#endif
}


#else // HEADER

#include "../Inputs/system-header-simulator-cxx.h"

class MySet {
  int *storage;
  unsigned size;
public:
  MySet() : storage(0), size(0) {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
  }

  MySet(unsigned n) : storage(new int[n]), size(n) {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
  }

  ~MySet() { delete[] storage; }

  bool isEmpty() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return size == 0;
  }

  struct iterator {
    int *impl;

    iterator(int *p) : impl(p) {}
  };

  iterator begin() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return iterator(storage);
  }

  iterator end() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return iterator(storage+size);
  }

  typedef int *raw_iterator;

  raw_iterator raw_begin() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return storage;
  }
  raw_iterator raw_end() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return storage + size;
  }
};

class MySetSubclass : public MySet {
public:
  MySetSubclass() {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
  }

  void useIterator(iterator i) {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
  }
};

class BeginOnlySet {
  MySet impl;
public:
  struct IterImpl {
    MySet::iterator impl;
    typedef std::forward_iterator_tag iterator_category;

    IterImpl(MySet::iterator i) : impl(i) {
      clang_analyzer_checkInlined(true);
#if INLINE
      // expected-warning@-2 {{TRUE}}
#endif
    }
  };

  BeginOnlySet() {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
  }

  typedef IterImpl wrapped_iterator;

  wrapped_iterator begin() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return IterImpl(impl.begin());
  }
};

class IteratorTypedefOnlySet {
  MySet impl;
public:

  IteratorTypedefOnlySet() {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
  }

  typedef MySet::iterator iterator;

  iterator start() {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
    return impl.begin();
  }
};

class IteratorUsingOnlySet {
  MySet impl;
public:

  IteratorUsingOnlySet() {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
  }

  using iterator = MySet::iterator;

  iterator start() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return impl.begin();
  }
};

class IteratorStructOnlySet {
  MySet impl;
public:

  IteratorStructOnlySet() {
    clang_analyzer_checkInlined(true);
#if INLINE
    // expected-warning@-2 {{TRUE}}
#endif
  }

  struct iterator {
    int *impl;
  };

  iterator start() {
    clang_analyzer_checkInlined(true);
    #if INLINE
        // expected-warning@-2 {{TRUE}}
    #endif
    return iterator{impl.begin().impl};
  }
};

#endif // HEADER
