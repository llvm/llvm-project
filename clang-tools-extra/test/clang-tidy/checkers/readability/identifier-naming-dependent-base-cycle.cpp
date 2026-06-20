// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.AggressiveDependentMemberLookup: true \
// RUN:   }}' -- -fno-delayed-template-parsing

template <class T>
struct A;

template <class T>
struct A<const T> {
  int x;
};

template <class T>
struct A : A<const T> {
  A() { this->x; }
};
