// RUN: %clang_cc1 -std=c++11 %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -include-pch %t.pch -verify

// RUN: %clang_cc1 -std=c++11 %s -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -include-pch %t.pch -verify

// expected-no-diagnostics

// rdar://12631281
// This reduced test case exposed a use-after-free memory bug, which was reliable
// reproduced only on guarded malloc (and probably valgrind).

#ifndef HEADER
#define HEADER

template < class _T2> struct  is_convertible;
template <> struct is_convertible<int> { typedef int type; };

template <class _T1, class _T2> struct  pair {
  typedef _T1 first_type;
  typedef _T2 second_type;
  template <class _U1, class _U2, class = typename is_convertible< first_type>::type>
    pair(_U1&& , _U2&& ); // expected-note {{candidate}}
};

template <class _ForwardIterator>
pair<_ForwardIterator, _ForwardIterator> __equal_range(_ForwardIterator) {
  return pair<_ForwardIterator, _ForwardIterator>(0, 0); // expected-error {{no matching constructor}}
}

template <class _ForwardIterator>
pair<_ForwardIterator, _ForwardIterator> equal_range( _ForwardIterator a) {
  return __equal_range(a); // expected-note {{instantiation}}
}

class A {
  pair<int, int> range() {
    return equal_range(0); // expected-note {{instantiation}}
  }
};

#else

#endif
