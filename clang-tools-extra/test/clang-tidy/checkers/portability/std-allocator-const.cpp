// RUN: %check_clang_tidy -std=c++11-or-later %s portability-std-allocator-const %t -- -- -fno-delayed-template-parsing

#include <deque>
#include <forward_list>
#include <list>
#include <set>
#include <stack>
#include <unordered_set>
#include <vector>

namespace absl {
template <class K, class H = std::hash<K>, class Eq = std::equal_to<K>, class A = std::allocator<K>>
class flat_hash_set {};
} // namespace absl

template <class T>
class allocator {};

void simple(const std::vector<const char> &v, std::deque<const short> *d) {
  // CHECK-MESSAGES: [[#@LINE-1]]:19: warning: container using std::allocator<const T> is a deprecated libc++ extension; remove const for compatibility with other standard libraries
  // CHECK-MESSAGES: [[#@LINE-2]]:47: warning: container
  std::list<const long> l;
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container

  std::multiset<int *const> ms;
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container
  std::set<const std::hash<int>> s;
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container
  std::unordered_multiset<int *const> ums;
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container
  std::unordered_set<const int> us;
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container

  absl::flat_hash_set<const int> fhs;
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container

  using my_vector = std::vector<const int>;
  // CHECK-MESSAGES: [[#@LINE-1]]:21: warning: container
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
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container

  std::vector<T> neg1;
  std::forward_list<const T> neg2;
}
void use_temp1() { temp1<int>(); }

template <class T>
void temp2() {
  // Match std::vector<const dependent> for the uninstantiated temp2.
  std::vector<const T> v;
  // CHECK-MESSAGES: [[#@LINE-1]]:3: warning: container

  std::vector<T> neg1;
  std::forward_list<const T> neg2;
}
