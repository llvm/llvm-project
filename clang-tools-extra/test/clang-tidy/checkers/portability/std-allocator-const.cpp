// RUN: %check_clang_tidy -std=c++11-or-later %s portability-std-allocator-const %t -- -- -fno-delayed-template-parsing

namespace std {
typedef unsigned size_t;

template <class T>
class allocator {};
template <class T>
class hash {};
template <class T>
class equal_to {};
template <class T>
class less {};

template <class T, class A = std::allocator<T>>
class deque {};
template <class T, class A = std::allocator<T>>
class forward_list {};
template <class T, class A = std::allocator<T>>
class list {};
template <class T, class A = std::allocator<T>>
class vector {};

template <class K, class C = std::less<K>, class A = std::allocator<K>>
class multiset {};
template <class K, class C = std::less<K>, class A = std::allocator<K>>
class set {};
template <class K, class H = std::hash<K>, class Eq = std::equal_to<K>, class A = std::allocator<K>>
class unordered_multiset {};
template <class K, class H = std::hash<K>, class Eq = std::equal_to<K>, class A = std::allocator<K>>
class unordered_set {};

template <class T, class C = std::deque<T>>
class stack {};
} // namespace std

namespace absl {
template <class K, class H = std::hash<K>, class Eq = std::equal_to<K>, class A = std::allocator<K>>
class flat_hash_set {};
} // namespace absl

template <class T>
class allocator {};

void simple(const std::vector<const char> &v, std::deque<const short> *d) {
  // CHECK-MESSAGES: [[#@LINE-1]]:24: warning: container using std::allocator<const T> is a deprecated libc++ extension; remove const for compatibility with other standard libraries
  // CHECK-MESSAGES: [[#@LINE-2]]:52: warning: container
  std::list<const long> l;
  // CHECK-MESSAGES: [[#@LINE-1]]:8: warning: container

  std::multiset<int *const> ms;
  // CHECK-MESSAGES: [[#@LINE-1]]:8: warning: container
  std::set<const std::hash<int>> s;
  // CHECK-MESSAGES: [[#@LINE-1]]:8: warning: container
  std::unordered_multiset<int *const> ums;
  // CHECK-MESSAGES: [[#@LINE-1]]:8: warning: container
  std::unordered_set<const int> us;
  // CHECK-MESSAGES: [[#@LINE-1]]:8: warning: container

  absl::flat_hash_set<const int> fhs;
  // CHECK-MESSAGES: [[#@LINE-1]]:9: warning: container

  using my_vector = std::vector<const int>;
  // CHECK-MESSAGES: [[#@LINE-1]]:26: warning: container
  my_vector v1;
  using my_vector2 = my_vector;

  std::vector<int> neg1;
  std::vector<const int *> neg2;                     // not const T
  std::vector<const int, allocator<const int>> neg3; // not use std::allocator<const T>
  std::allocator<const int> a;                       // not caught, but rare
  std::forward_list<const int> forward;              // not caught, but rare
  std::stack<const int> stack;                       // not caught, but rare
}

template <class T>
void temp1() {
  std::vector<const T> v;
  // CHECK-MESSAGES: [[#@LINE-1]]:8: warning: container

  std::vector<T> neg1;
  std::forward_list<const T> neg2;
}
void use_temp1() { temp1<int>(); }

template <class T>
void temp2() {
  // Match std::vector<const dependent> for the uninstantiated temp2.
  std::vector<const T> v;
  // CHECK-MESSAGES: [[#@LINE-1]]:8: warning: container

  std::vector<T> neg1;
  std::forward_list<const T> neg2;
}
