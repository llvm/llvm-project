// RUN: %check_clang_tidy -std=c++20 %s modernize-use-std-erase %t
#include <deque>
#include <list>
#include <string>
#include <vector>

namespace std {
template <class ForwardIt, class T>
ForwardIt remove(ForwardIt first, ForwardIt last, const T& value);

template <class ForwardIt, class UnaryPredicate>
ForwardIt remove_if(ForwardIt first, ForwardIt last, UnaryPredicate p);

// Dummy implementation
template <class ForwardIt, class UnaryPredicate>
ForwardIt remove_if(ForwardIt first, ForwardIt last, UnaryPredicate p) {
  return first;
}

} // namespace std

// Custom container - should be ignored
template <typename T>
struct MyContainer {
  using iterator = T*;
  iterator begin();
  iterator end();
  iterator erase(iterator, iterator);
};

void test_standard_remove_idiom() {
  std::vector<int> v;
  v.erase(std::remove(v.begin(), v.end(), 42), v.end());
  // CHECK-MESSAGES: {{.*}}: warning: prefer std::erase over the erase-remove idiom [modernize-use-std-erase]
  // CHECK-FIXES: std::erase(v, 42);

  std::deque<int> d;
  d.erase(std::remove(d.begin(), d.end(), 42), d.end());
  // CHECK-MESSAGES: {{.*}}: warning: prefer std::erase over the erase-remove idiom [modernize-use-std-erase]
  // CHECK-FIXES: std::erase(d, 42);
}

void test_standard_remove_if_idiom() {
  std::vector<int> v;
  auto IsNegative = [](int x) { return x < 0; };
  
  v.erase(std::remove_if(v.begin(), v.end(), IsNegative), v.end());
  // CHECK-MESSAGES: {{.*}}: warning: prefer std::erase_if over the erase-remove_if idiom [modernize-use-std-erase]
  // CHECK-FIXES: std::erase_if(v, IsNegative);

  std::list<int> l;
  l.erase(std::remove_if(l.begin(), l.end(), IsNegative), l.end());
  // CHECK-MESSAGES: {{.*}}: warning: prefer std::erase_if over the erase-remove_if idiom [modernize-use-std-erase]
  // CHECK-FIXES: std::erase_if(l, IsNegative);
}

void test_string_special_case() {
  std::string s;
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
  // CHECK-MESSAGES: {{.*}}: warning: prefer std::erase over the erase-remove idiom [modernize-use-std-erase]
  // CHECK-FIXES: std::erase(s, ' ');
}

auto IsEven = [](int i) { return i % 2 == 0; };

void test_remove_negative_cases() {
  std::vector<int> v;
  std::vector<int> v2;

  v.erase(std::remove_if(v.rbegin(), v.rend(), IsEven), v.rend());
  // CHECK-FIXES: v.erase(std::remove_if(v.rbegin(), v.rend(), IsEven), v.rend());

  MyContainer<int> c;
  c.erase(std::remove_if(c.begin(), c.end(), IsEven), c.end());
  // CHECK-FIXES: c.erase(std::remove_if(c.begin(), c.end(), IsEven), c.end());

  v.erase(std::remove_if(v.begin() + 1, v.end(), IsEven), v.end());
  // CHECK-FIXES: v.erase(std::remove_if(v.begin() + 1, v.end(), IsEven), v.end());
 
  v.erase(std::remove(v2.begin(), v2.end(), 1), v.end());
  // CHECK-FIXES: v.erase(std::remove(v2.begin(), v2.end(), 1), v.end());

  v.erase(std::remove(v.begin(), v.end(), 1), v.end() - 1);
  // CHECK-FIXES: v.erase(std::remove(v.begin(), v.end(), 1), v.end() - 1);

  auto it = std::remove(v.begin(), v.end(), 1);
  v.erase(it, v.end());
  // CHECK-FIXES: v.erase(it, v.end());
}
