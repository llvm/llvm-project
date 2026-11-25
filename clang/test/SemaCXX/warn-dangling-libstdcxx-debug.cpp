// RUN: %clang_cc1 -std=c++17 -Wdangling-gsl -verify %s

namespace std {
namespace __cxx1998 {
template <class T> struct vector {
  T &at(int);
  struct iterator {};
  iterator begin();
};
inline namespace __cxx11 {
template <class T> struct list {
  T &front();
};
} // namespace __cxx11
} // namespace __cxx1998

inline namespace __debug {
template <class T> struct vector : __cxx1998::vector<T> {};
template <class T> struct list : __cxx1998::list<T> {};
} // namespace __debug
} // namespace std

void check(std::vector<int> &live) {
  auto &ref = std::vector<int>{}.at(0);
  // expected-warning@-1 {{object backing the pointer will be destroyed}}
  auto iter = std::vector<int>{}.begin();
  // expected-warning@-1 {{object backing the pointer will be destroyed}}
  auto &front = std::list<int>{}.front();
  // expected-warning@-1 {{object backing the pointer will be destroyed}}
  auto &ok = live.at(0);
  auto liveIter = live.begin();
  auto value = std::vector<int>{}.at(0);
}

namespace other::__cxx1998 {
struct vector {
  int &at(int);
};
} // namespace other::__cxx1998

namespace std::other {
struct vector {
  int &at(int);
};
} // namespace std::other

void checkUnrelated() {
  auto &ref = other::__cxx1998::vector{}.at(0);
  auto &nested = std::other::vector{}.at(0);
}
