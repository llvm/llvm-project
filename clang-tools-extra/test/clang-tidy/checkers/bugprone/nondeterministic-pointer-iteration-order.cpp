// RUN: %check_clang_tidy %s bugprone-nondeterministic-pointer-iteration-order %t -- -- -I%S -std=c++!4

#include "Inputs/system-header-simulator/sim_set"
#include "Inputs/system-header-simulator/sim_unordered_set"
#include "Inputs/system-header-simulator/sim_map"
#include "Inputs/system-header-simulator/sim_unordered_map"
#include "Inputs/system-header-simulator/sim_vector"
#include "Inputs/system-header-simulator/sim_algorithm"

template<class T>
void f(T x);

void PointerIteration() {
  int a = 1, b = 2;
  std::set<int> OrderedIntSet = {a, b};
  std::set<int *> OrderedPtrSet = {&a, &b};
  std::unordered_set<int> UnorderedIntSet = {a, b};
  std::unordered_set<int *> UnorderedPtrSet = {&a, &b};
  std::map<int, int> IntMap = { std::make_pair(a,a), std::make_pair(b,b) };
  std::map<int*, int*> PtrMap = { std::make_pair(&a,&a), std::make_pair(&b,&b) };
  std::unordered_map<int, int> IntUnorderedMap = { std::make_pair(a,a), std::make_pair(b,b) };
  std::unordered_map<int*, int*> PtrUnorderedMap = { std::make_pair(&a,&a), std::make_pair(&b,&b) };

  for (auto i : OrderedIntSet) // no-warning
    f(i);

  for (auto i : OrderedPtrSet) // no-warning
    f(i);

  for (auto i : UnorderedIntSet) // no-warning
    f(i);

  for (auto i : UnorderedPtrSet)
    f(i);
  // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: iteration of pointers is nondeterministic

  for (auto &i : UnorderedPtrSet)
    f(i);
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: iteration of pointers is nondeterministic

  for (auto &i : IntMap) // no-warning
    f(i);

  for (auto &i : PtrMap) // no-warning
    f(i);

  for (auto &i : IntUnorderedMap) // no-warning
    f(i);

  for (auto &i : PtrUnorderedMap)
    f(i);
  // CHECK-MESSAGES: :[[@LINE-2]]:18: warning: iteration of pointers is nondeterministic
}

bool g (int *x) { return true; }
bool h (int x) { return true; }

void PointerSorting() {
  int a = 1, b = 2, c = 3;
  std::vector<int> V1 = {a, b};
  std::vector<int *> V2 = {&a, &b};

  std::is_sorted(V1.begin(), V1.end());                    // no-warning
  std::nth_element(V1.begin(), V1.begin() + 1, V1.end());  // no-warning
  std::partial_sort(V1.begin(), V1.begin() + 1, V1.end()); // no-warning
  std::sort(V1.begin(), V1.end());                         // no-warning
  std::stable_sort(V1.begin(), V1.end());                  // no-warning
  std::partition(V1.begin(), V1.end(), h);                 // no-warning
  std::stable_partition(V1.begin(), V1.end(), h);          // no-warning
  std::is_sorted(V2.begin(), V2.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: sorting pointers is nondeterministic
  std::nth_element(V2.begin(), V2.begin() + 1, V2.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: sorting pointers is nondeterministic
  std::partial_sort(V2.begin(), V2.begin() + 1, V2.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: sorting pointers is nondeterministic
  std::sort(V2.begin(), V2.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: sorting pointers is nondeterministic
  std::stable_sort(V2.begin(), V2.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: sorting pointers is nondeterministic
  std::partition(V2.begin(), V2.end(), g);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: sorting pointers is nondeterministic
  std::stable_partition(V2.begin(), V2.end(), g);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: sorting pointers is nondeterministic
}
