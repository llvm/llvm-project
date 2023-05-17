// RUN: %check_clang_tidy %s google-build-explicit-make-pair %t

namespace std {
template <class T1, class T2>
struct pair {
  pair(T1 x, T2 y) {}
};

template <class T1, class T2>
pair<T1, T2> make_pair(T1 x, T2 y) {
  return pair<T1, T2>(x, y);
}
}

template <typename T>
void templ(T a, T b) {
  std::make_pair<T, unsigned>(a, b);
  std::make_pair<int, int>(1, 2);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: for C++11-compatibility, omit template arguments from make_pair
// CHECK-FIXES: std::make_pair(1, 2)
}

template <typename T>
int t();

void test(int i) {
  std::make_pair<int, int>(i, i);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: for C++11-compatibility, omit template arguments from make_pair
// CHECK-FIXES: std::make_pair(i, i)

  std::make_pair<unsigned, int>(i, i);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: for C++11-compatibility, use pair directly
// CHECK-FIXES: std::pair<unsigned, int>(i, i)

  std::make_pair<int, unsigned>(i, i);
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: for C++11-compatibility, use pair directly
// CHECK-FIXES: std::pair<int, unsigned>(i, i)

#define M std::make_pair<int, unsigned>(i, i);
M
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: for C++11-compatibility, use pair directly
// Can't fix in macros.
// CHECK-FIXES: #define M std::make_pair<int, unsigned>(i, i);
// CHECK-FIXES-NEXT: M

  templ(i, i);
  templ(1U, 2U);

  std::make_pair(i, 1); // no-warning
  std::make_pair(t<int>, 1);
}
