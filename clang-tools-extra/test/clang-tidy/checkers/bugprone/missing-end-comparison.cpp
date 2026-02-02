// RUN: %check_clang_tidy %s bugprone-missing-end-comparison %t -- -- -std=c++17

namespace std {
  template<typename T> struct iterator_traits;
  struct forward_iterator_tag {};
  
  typedef long int ptrdiff_t;
  typedef decltype(nullptr) nullptr_t;

  template<typename T>
  struct vector {
    typedef T* iterator;
    typedef const T* const_iterator;
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
  };

  template<class InputIt, class T>
  InputIt find(InputIt first, InputIt last, const T& value);

  template<class ForwardIt, class T>
  ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value);

  template<class ForwardIt, class ForwardIt2>
  ForwardIt search(ForwardIt first, ForwardIt last, ForwardIt first2, ForwardIt2 last2);

  template<class ForwardIt>
  ForwardIt min_element(ForwardIt first, ForwardIt last);
}

struct CustomIterator {
    int* ptr;
    using iterator_category = std::forward_iterator_tag;
    using value_type = int;
    using difference_type = long;
    using pointer = int*;
    using reference = int&;

    int& operator*() const { return *ptr; }
    CustomIterator& operator++() { ++ptr; return *this; }
    bool operator==(const CustomIterator& other) const { return ptr == other.ptr; }
    bool operator!=(const CustomIterator& other) const { return ptr != other.ptr; }
    
    explicit operator bool() const { return ptr != nullptr; }
};

void test_raw_pointers() {
  int arr[] = {1, 2, 3};
  int* begin = arr;
  int* end = arr + 3;

  if (std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if (std::find(begin, end, 2) != end) {}

  while (std::lower_bound(begin, end, 2)) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of standard algorithm used in boolean context
  // CHECK-FIXES: while (std::lower_bound(begin, end, 2) != end) { break; }

  if (std::find(begin, end, 2) != end) {}
}

void test_vector() {
  std::vector<int> v;
  if (std::find(v.begin(), v.end(), 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context
  // CHECK-FIXES: if (std::find(v.begin(), v.end(), 2) != v.end()) {}

  if (std::min_element(v.begin(), v.end())) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context
  // CHECK-FIXES: if (std::min_element(v.begin(), v.end()) != v.end()) {}
}

void test_custom_iterator() {
  CustomIterator begin{nullptr}, end{nullptr};
  if (std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context
  // CHECK-FIXES: if (std::find(begin, end, 2) != end) {}
}

void test_complex_end() {
  int arr[] = {1, 2, 3};
  if (std::find(arr, arr + 3, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context
  // CHECK-FIXES: if (std::find(arr, arr + 3, 2) != arr + 3) {}
}

void test_sentinel() {
  int* ptr = nullptr;
  if (std::find<int*>(ptr, nullptr, 10)) {}
  // No warning expected for nullptr sentinel
}