// RUN: %check_clang_tidy -std=c++11,c++14 %s modernize-use-auto %t -- -- -I %S/Inputs/modernize-use-auto
// FIXME: Fix the checker to work in C++17 mode.

#include "containers.h"

void f_array() {
  std::array<int, 4> C;
  std::array<int, 4>::iterator ArrayI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators [modernize-use-auto]
  // CHECK-FIXES: auto ArrayI1 = C.begin();

  std::array<int, 5>::reverse_iterator ArrayI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto ArrayI2 = C.rbegin();

  const std::array<int, 3> D;
  std::array<int, 3>::const_iterator ArrayI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto ArrayI3 = D.begin();

  std::array<int, 5>::const_reverse_iterator ArrayI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto ArrayI4 = D.rbegin();
}

void f_deque() {
  std::deque<int> C;
  std::deque<int>::iterator DequeI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto DequeI1 = C.begin();

  std::deque<int>::reverse_iterator DequeI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto DequeI2 = C.rbegin();

  const std::deque<int> D;
  std::deque<int>::const_iterator DequeI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto DequeI3 = D.begin();

  std::deque<int>::const_reverse_iterator DequeI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto DequeI4 = D.rbegin();
}

void f_forward_list() {
  std::forward_list<int> C;
  std::forward_list<int>::iterator FListI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto FListI1 = C.begin();

  const std::forward_list<int> D;
  std::forward_list<int>::const_iterator FListI2 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto FListI2 = D.begin();
}

void f_list() {
  std::list<int> C;
  std::list<int>::iterator ListI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto ListI1 = C.begin();
  std::list<int>::reverse_iterator ListI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto ListI2 = C.rbegin();

  const std::list<int> D;
  std::list<int>::const_iterator ListI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto ListI3 = D.begin();
  std::list<int>::const_reverse_iterator ListI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto ListI4 = D.rbegin();
}

void f_vector() {
  std::vector<int> C;
  std::vector<int>::iterator VecI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto VecI1 = C.begin();

  std::vector<int>::reverse_iterator VecI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto VecI2 = C.rbegin();

  const std::vector<int> D;
  std::vector<int>::const_iterator VecI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto VecI3 = D.begin();

  std::vector<int>::const_reverse_iterator VecI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto VecI4 = D.rbegin();
}

void f_map() {
  std::map<int, int> C;
  std::map<int, int>::iterator MapI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MapI1 = C.begin();

  std::map<int, int>::reverse_iterator MapI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MapI2 = C.rbegin();

  const std::map<int, int> D;
  std::map<int, int>::const_iterator MapI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MapI3 = D.begin();

  std::map<int, int>::const_reverse_iterator MapI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MapI4 = D.rbegin();
}

void f_multimap() {
  std::multimap<int, int> C;
  std::multimap<int, int>::iterator MMapI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MMapI1 = C.begin();

  std::multimap<int, int>::reverse_iterator MMapI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MMapI2 = C.rbegin();

  const std::multimap<int, int> D;
  std::multimap<int, int>::const_iterator MMapI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MMapI3 = D.begin();

  std::multimap<int, int>::const_reverse_iterator MMapI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MMapI4 = D.rbegin();
}

void f_set() {
  std::set<int> C;
  std::set<int>::iterator SetI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto SetI1 = C.begin();

  std::set<int>::reverse_iterator SetI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto SetI2 = C.rbegin();

  const std::set<int> D;
  std::set<int>::const_iterator SetI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto SetI3 = D.begin();

  std::set<int>::const_reverse_iterator SetI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto SetI4 = D.rbegin();
}

void f_multiset() {
  std::multiset<int> C;
  std::multiset<int>::iterator MSetI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MSetI1 = C.begin();

  std::multiset<int>::reverse_iterator MSetI2 = C.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MSetI2 = C.rbegin();

  const std::multiset<int> D;
  std::multiset<int>::const_iterator MSetI3 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MSetI3 = D.begin();

  std::multiset<int>::const_reverse_iterator MSetI4 = D.rbegin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto MSetI4 = D.rbegin();
}

void f_unordered_map() {
  std::unordered_map<int, int> C;
  std::unordered_map<int, int>::iterator UMapI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto UMapI1 = C.begin();

  const std::unordered_map<int, int> D;
  std::unordered_map<int, int>::const_iterator UMapI2 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto UMapI2 = D.begin();
}

void f_unordered_multimap() {
  std::unordered_multimap<int, int> C;
  std::unordered_multimap<int, int>::iterator UMMapI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto UMMapI1 = C.begin();

  const std::unordered_multimap<int, int> D;
  std::unordered_multimap<int, int>::const_iterator UMMapI2 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto UMMapI2 = D.begin();
}

void f_unordered_set() {
  std::unordered_set<int> C;
  std::unordered_set<int>::iterator USetI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto USetI1 = C.begin();

  const std::unordered_set<int> D;
  std::unordered_set<int>::const_iterator USetI2 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto USetI2 = D.begin();
}

void f_unordered_multiset() {
  std::unordered_multiset<int> C;
  std::unordered_multiset<int>::iterator UMSetI1 = C.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto UMSetI1 = C.begin();

  const std::unordered_multiset<int> D;
  std::unordered_multiset<int>::const_iterator UMSetI2 = D.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto UMSetI2 = D.begin();
}

typedef std::vector<int>::iterator int_iterator;

std::vector<int> Vec;
std::unordered_map<int, int> Map;

void sugar() {
  // Types with more sugar should work. Types with less should not.
  int_iterator more_sugar = Vec.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto more_sugar = Vec.begin();
}

void initializer_list() {
  // Initialization from initializer lists isn't allowed. Using 'auto' would
  // result in std::initializer_list being deduced for the type.
  std::unordered_map<int, int>::iterator I{Map.begin()};
  std::unordered_map<int, int>::iterator I2 = {Map.begin()};
}

void construction() {
  // Various forms of construction. Default constructors and constructors with
  // all-default parameters shouldn't get transformed. Construction from other
  // types is also not allowed.

  std::unordered_map<int, int>::iterator copy(Map.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto copy(Map.begin());

  std::unordered_map<int, int>::iterator def;
  std::unordered_map<int, int>::const_iterator constI;

  // Implicit conversion.
  std::unordered_map<int, int>::const_iterator constI2 = def;
  std::unordered_map<int, int>::const_iterator constI3(def);

  // Explicit conversion
  std::unordered_map<int, int>::const_iterator constI4
      = std::unordered_map<int, int>::const_iterator(def);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto constI4
  // CHECK-FIXES-NEXT: = std::unordered_map<int, int>::const_iterator(def);
}

void pointer_to_iterator() {
  int_iterator I = Vec.begin();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto I = Vec.begin();

  // Pointers and references to iterators are not transformed.
  int_iterator *IPtr = &I;
  int_iterator &IRef = I;
}

void loop() {
  for (std::vector<int>::iterator I = Vec.begin(); I != Vec.end(); ++I) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use auto when declaring iterators
    // CHECK-FIXES: for (auto I = Vec.begin(); I != Vec.end(); ++I)
  }

  for (int_iterator I = Vec.begin(), E = Vec.end(); I != E; ++I) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use auto when declaring iterators
    // CHECK-FIXES: for (auto I = Vec.begin(), E = Vec.end(); I != E; ++I)
  }

  std::vector<std::vector<int>::iterator> IterVec;
  for (std::vector<int>::iterator I : IterVec) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use auto when declaring iterators
    // CHECK-FIXES: for (auto I : IterVec)
  }
}

void cv_qualifiers() {
  // Make sure references and cv qualifiers don't get removed (i.e. replaced
  // with just 'auto').
  const auto & I = Vec.begin();
  auto && I2 = Vec.begin();
}

void cleanup() {
  // Passing a string as an argument to introduce a temporary object that will
  // create an expression with cleanups.
  std::map<std::string, int> MapFind;
  std::map<std::string, int>::iterator I = MapFind.find("foo");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto I = MapFind.find("foo");
}

void declaration_lists() {
  // Declaration lists that match the declaration type with written no-list
  // initializer are transformed.
  std::vector<int>::iterator I = Vec.begin(), E = Vec.end();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use auto when declaring iterators
  // CHECK-FIXES: auto I = Vec.begin(), E = Vec.end();

  // Declaration lists with non-initialized variables should not be transformed.
  std::vector<int>::iterator J = Vec.begin(), K;
}
