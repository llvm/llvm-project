// RUN: %check_clang_tidy -std=c++11-or-later %s misc-function-pointer-equality %t

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

void f();
void g();

void test_containers_std() {
  std::set<void (*)()> a1;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: associative or unordered container 'a1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::multiset<void (*)()> b1;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: associative or unordered container 'b1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::unordered_set<void (*)()> c1;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: associative or unordered container 'c1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::unordered_multiset<void (*)()> d1;
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: associative or unordered container 'd1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::map<void (*)(), int> e1;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: associative or unordered container 'e1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::multimap<void (*)(), int> f1;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: associative or unordered container 'f1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::unordered_map<void (*)(), int> g1;
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: associative or unordered container 'g1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::unordered_multimap<void (*)(), int> h1;
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: associative or unordered container 'h1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  const std::set<void (*)()> i1;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: associative or unordered container 'i1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  
  // Test `using` directive to rename a type.
  using FnMap = std::unordered_map<void (*)(), int>;
  FnMap j1;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: associative or unordered container 'j1' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]

  // Should not lint (either because they don't have a function pointer key, or are not actually a container).
  std::map<int, void (*)()> k1;
  std::unordered_map<int, void (*)()> l1;
  const std::set<void (*)()> &my_ref = i1;
  const std::set<void (*)()> *my_ptr = &i1;
}

namespace std
{
void test_containers_namespace() {
  set<void (*)()> s3;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: associative or unordered container 's3' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  unordered_set<void (*)()> s4;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: associative or unordered container 's4' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  map<void (*)(), int> m3;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: associative or unordered container 'm3' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  unordered_map<void (*)(), int> m4;
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: associative or unordered container 'm4' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]

  vector<void (*)()> v2;
  map<int, void (*)()> m_val2;
}
}

struct S {
  std::set<void (*)()> member;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: associative or unordered container 'member' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
};

std::set<void (*)()> global_set;
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: associative or unordered container 'global_set' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]

void test_param(std::set<void (*)()> param_set) {}
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: associative or unordered container 'param_set' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]

void test_unnamed_param(std::set<void (*)()>);
// CHECK-MESSAGES: :[[@LINE-1]]:45: warning: associative or unordered container with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]

std::set<void (*)()> returns_set();

template <typename T>
bool generic_eq(T a, T b) {
  return a == b;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
}

template <typename T>
void generic_set() {
  std::set<T> s;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: associative or unordered container 's' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
}

template <typename T>
void non_dependent_in_template() {
  bool b = (f == g);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: function pointer equality comparison is unsound with '--icf=all' enabled [misc-function-pointer-equality]
  std::set<void (*)()> s;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: associative or unordered container 's' with a function pointer key type is unsound with '--icf=all' enabled [misc-function-pointer-equality]
}

void test_templates() {
  generic_eq(f, g);
  generic_set<void (*)()>();
  non_dependent_in_template<int>();
  non_dependent_in_template<double>();
}
