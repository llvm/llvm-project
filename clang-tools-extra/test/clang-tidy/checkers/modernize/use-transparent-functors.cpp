// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-use-transparent-functors %t

#include <functional>
#include <memory>

namespace std {

template <
    class Key,
    class Compare = std::less<>,
    class Allocator = std::allocator<Key>>
class set {};

template <
    class Key,
    class Compare = std::less<Key>,
    class Allocator = std::allocator<Key>>
class set2 {};

template <class InputIt, class UnaryPredicate>
InputIt find_if(InputIt first, InputIt last,
                UnaryPredicate p);

template <class RandomIt, class Compare>
void sort(RandomIt first, RandomIt last, Compare comp);

class iterator {};
class string {};
}

int main() {
  using std::set;
  using std::less;
  std::set<int, std::less<int>> s;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: prefer transparent functors 'less<>' [modernize-use-transparent-functors]
  // CHECK-FIXES: std::set<int, std::less<>> s;
  set<int, std::less<int>> s2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: prefer transparent functors
  // CHECK-FIXES: set<int, std::less<>> s2;
  set<int, less<int>> s3;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: prefer transparent functors
  // CHECK-FIXES: set<int, less<>> s3;
  std::set<int, std::less<>> s4;
  std::set<char *, std::less<std::string>> s5;
  std::set<set<int, less<int>>, std::less<>> s6;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: prefer transparent functors
  // CHECK-FIXES: std::set<set<int, less<>>, std::less<>> s6;
  std::iterator begin, end;
  sort(begin, end, std::less<int>());
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: prefer transparent functors
  std::sort(begin, end, std::less<>());
  find_if(begin, end, std::logical_not<bool>());
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: prefer transparent functors
  std::find_if(begin, end, std::logical_not<>());
  using my_set = std::set<int, std::less<int>>;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: prefer transparent functors
  // CHECK-FIXES: using my_set = std::set<int, std::less<>>;
  using my_set2 = std::set<char*, std::less<std::string>>;
  using my_less = std::less<std::string>;
  find_if(begin, end, my_less());
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: prefer transparent functors
  std::set2<int> control;
}

struct ImplicitTypeLoc : std::set2<std::less<int>> {
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: prefer transparent functors
  ImplicitTypeLoc() {}
};
