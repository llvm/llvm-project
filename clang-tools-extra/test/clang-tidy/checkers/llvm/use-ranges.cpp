// RUN: %check_clang_tidy %s llvm-use-ranges %t

// Test that the header is included
// CHECK-FIXES: #include "llvm/ADT/STLExtras.h"

namespace std {

template <typename T> class vector {
public:
  using iterator = T *;
  using const_iterator = const T *;

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;
};

template <typename T> T* begin(T (&arr)[5]);
template <typename T> T* end(T (&arr)[5]);

template <class InputIt, class T>
InputIt find(InputIt first, InputIt last, const T &value);

template <class RandomIt>
void sort(RandomIt first, RandomIt last);

template <class RandomIt>
void stable_sort(RandomIt first, RandomIt last);

template <class InputIt, class UnaryPredicate>
bool all_of(InputIt first, InputIt last, UnaryPredicate p);

template <class InputIt, class UnaryFunction>
UnaryFunction for_each(InputIt first, InputIt last, UnaryFunction f);

template <class ForwardIt, class T>
ForwardIt remove(ForwardIt first, ForwardIt last, const T& value);

template <class ForwardIt>
ForwardIt min_element(ForwardIt first, ForwardIt last);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

template <class InputIt1, class InputIt2>
bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2);

template <class InputIt, class OutputIt>
OutputIt copy(InputIt first, InputIt last, OutputIt d_first);

template <class ForwardIt, class T>
void fill(ForwardIt first, ForwardIt last, const T& value);

template <class BidirIt>
void reverse(BidirIt first, BidirIt last);

template <class ForwardIt>
ForwardIt unique(ForwardIt first, ForwardIt last);

template <class ForwardIt>
bool is_sorted(ForwardIt first, ForwardIt last);

template <class InputIt1, class InputIt2>
bool includes(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2);

} // namespace std

bool is_even(int x);
void double_ref(int& x);

void test_positive() {
  std::vector<int> vec;
  int arr[5] = {1, 2, 3, 4, 5};
  
  auto it1 = std::find(vec.begin(), vec.end(), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: auto it1 = llvm::find(vec, 3);

  auto it2 = std::find(std::begin(arr), std::end(arr), 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: auto it2 = llvm::find(arr, 3);

  std::stable_sort(vec.begin(), vec.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: llvm::stable_sort(vec);

  bool all = std::all_of(vec.begin(), vec.end(), is_even);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: bool all = llvm::all_of(vec, is_even);

  std::for_each(vec.begin(), vec.end(), double_ref);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: llvm::for_each(vec, double_ref);

  auto min_it = std::min_element(vec.begin(), vec.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: auto min_it = llvm::min_element(vec);

  std::vector<int> vec2;
  bool eq = std::equal(vec.begin(), vec.end(), vec2.begin(), vec2.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: bool eq = llvm::equal(vec, vec2);

  std::copy(vec.begin(), vec.end(), vec2.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: llvm::copy(vec, vec2.begin());

  std::fill(vec.begin(), vec.end(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: llvm::fill(vec, 0);
  
  auto last = std::unique(vec.begin(), vec.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: auto last = llvm::unique(vec);

  bool sorted = std::is_sorted(vec.begin(), vec.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: bool sorted = llvm::is_sorted(vec);

  std::includes(vec.begin(), vec.end(), std::begin(arr), std::end(arr));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use an LLVM range-based algorithm
  // CHECK-FIXES: llvm::includes(vec, arr);
}

void test_negative() {
  std::vector<int> v;
  
  // can not use `llvm::sort` because of potential different ordering from `std::sort`.
  std::sort(v.begin(), v.end());

  //non-begin/end iterators
  auto it1 = std::find(v.begin() + 1, v.end(), 2);
  auto it2 = std::find(v.begin(), v.end() - 1, 2);
  
  // Using different containers (3-arg equal)
  std::vector<int> v2;
  bool eq = std::equal(v.begin(), v.end(), v2.begin());
}
