// RUN: %check_clang_tidy -std=c++20 %s bugprone-missing-end-comparison %t

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

  namespace execution {
    struct sequenced_policy {};
    struct parallel_policy {};
    inline constexpr sequenced_policy seq;
    inline constexpr parallel_policy par;
  }

  template<class ExecutionPolicy, class InputIt, class T>
  InputIt find(ExecutionPolicy&& policy, InputIt first, InputIt last, const T& value);

  template<class ExecutionPolicy, class ForwardIt, class T>
  ForwardIt lower_bound(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value);

  template<class ForwardIt, class T>
  ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value);

  template<class ForwardIt, class ForwardIt2>
  ForwardIt search(ForwardIt first, ForwardIt last, ForwardIt first2, ForwardIt2 last2);

  template<class ForwardIt>
  ForwardIt min_element(ForwardIt first, ForwardIt last);

  template<class InputIt1, class InputIt2>
  struct pair {
    InputIt1 first;
    InputIt2 second;
  };

  namespace ranges {
    template<typename T>
    void* begin(T& t);
    template<typename T>
    void* end(T& t);

    struct FindFn {
      template<typename Range, typename T>
      void* operator()(Range&& r, const T& value) const;

      template<typename I, typename S, typename T>
      void* operator()(I first, S last, const T& value) const;
    };
    inline constexpr FindFn find;

    struct FindFirstOfFn {
      template<typename R1, typename R2>
      void* operator()(R1&& r1, R2&& r2) const;
      template<typename I1, typename S1, typename I2, typename S2>
      void* operator()(I1 f1, S1 l1, I2 f2, S2 l2) const;
    };
    inline constexpr FindFirstOfFn find_first_of;

    struct AdjacentFindFn {
      template<typename R>
      void* operator()(R&& r) const;
      template<typename I, typename S>
      void* operator()(I f, S l) const;
    };
    inline constexpr AdjacentFindFn adjacent_find;

    struct IsSortedUntilFn {
      template<typename R>
      void* operator()(R&& r) const;
      template<typename I, typename S>
      void* operator()(I f, S l) const;
    };
    inline constexpr IsSortedUntilFn is_sorted_until;
  }
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
  // CHECK-FIXES: if ((std::find(begin, end, 2) != end)) {}

  while (std::lower_bound(begin, end, 2)) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: while ((std::lower_bound(begin, end, 2) != end)) { break; }

  if (!std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(begin, end, 2) == end)) {}
}

void test_vector() {
  std::vector<int> v;
  if (std::find(v.begin(), v.end(), 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(v.begin(), v.end(), 2) != v.end())) {}

  if (std::min_element(v.begin(), v.end())) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::min_element(v.begin(), v.end()) != v.end())) {}
}

void test_variable_tracking() {
  int arr[] = {1, 2, 3};
  auto it = std::find(arr, arr + 3, 2);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != arr + 3)) {}

  if (!it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it == arr + 3)) {}
}

void test_ranges() {
  std::vector<int> v;
  if (std::ranges::find(v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(v, 2) != std::ranges::end(v))) {}

  auto it = std::ranges::find(v, 2);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != std::ranges::end(v))) {}

  std::vector<int> v1, v2;
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;

  if (std::ranges::find_first_of(v1, v2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find_first_of(v1, v2) != std::ranges::end(v1))) {}

  if (std::ranges::find_first_of(begin, end, begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find_first_of(begin, end, begin, end) != end)) {}

  if (std::ranges::adjacent_find(v1)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::adjacent_find(v1) != std::ranges::end(v1))) {}

  if (std::ranges::adjacent_find(begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::adjacent_find(begin, end) != end)) {}

  if (std::ranges::is_sorted_until(v1)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::is_sorted_until(v1) != std::ranges::end(v1))) {}

  if (std::ranges::is_sorted_until(begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::is_sorted_until(begin, end) != end)) {}
}

void test_ranges_iterator_pair() {
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;
  if (std::ranges::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(begin, end, 2) != end)) {}
}

void test_side_effects() {
  std::vector<int> get_vec();
  if (std::ranges::find(get_vec(), 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}

void test_negative() {
  std::vector<int> v;
  if (std::find(v.begin(), v.end(), 2) != v.end()) {}
  if (std::ranges::find(v, 2) == std::ranges::end(v)) {}
  auto it = std::find(v.begin(), v.end(), 2);
}

void test_nested_parens() {
  int arr[] = {1, 2, 3};
  if (!((std::find(arr, arr + 3, 2)))) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((((std::find(arr, arr + 3, 2))) == arr + 3)) {}
}

struct Data { std::vector<int> v; };
void test_member_expr(Data& d) {
  if (std::ranges::find(d.v, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::ranges::find(d.v, 2) != std::ranges::end(d.v))) {}
}

void test_nullptr_comparison() {
  if (std::find((int*)nullptr, (int*)nullptr, 2)) {}
}

void test_double_negation() {
  int arr[] = {1, 2, 3};
  auto it = std::find(arr, arr + 3, 2);
  if (!!it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if (!(it == arr + 3)) {}
}

void test_custom_iterator() {
  CustomIterator begin{nullptr}, end{nullptr};
  if (std::find(begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context
  // CHECK-FIXES: if ((std::find(begin, end, 2) != end)) {}
}

void test_search() {
  int arr[] = {1, 2, 3};
  int* begin = arr;
  int* end = arr + 3;

  if (std::search(begin, end, begin, end)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context
  // CHECK-FIXES: if ((std::search(begin, end, begin, end) != end)) {}
}

namespace other {
  bool find(int* b, int* e, int v);
}

void test_other_namespace() {
  int arr[] = {1};
  if (other::find(arr, arr + 1, 1)) {}
}

void test_execution_policy() {
  int arr[] = {1, 2, 3};
  int* begin = arr;
  int* end = arr + 3;

  if (std::find(std::execution::seq, begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(std::execution::seq, begin, end, 2) != end)) {}

  if (std::find(std::execution::par, begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::find(std::execution::par, begin, end, 2) != end)) {}

  auto it = std::find(std::execution::seq, begin, end, 2);
  if (it) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((it != end)) {}

  if (std::lower_bound(std::execution::seq, begin, end, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: if ((std::lower_bound(std::execution::seq, begin, end, 2) != end)) {}
}

void test_loops() {
  int arr[] = {1, 2, 3};
  int *begin = arr;
  int *end = arr + 3;

  while (std::find(begin, end, 2)) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: while ((std::find(begin, end, 2) != end)) { break; }

  do { } while (std::find(begin, end, 2));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: do { } while ((std::find(begin, end, 2) != end));

  for (auto it = std::find(begin, end, 2); it; ) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: for (auto it = std::find(begin, end, 2); (it != end); ) { break; }

  std::vector<int> v;
  for (auto it = std::ranges::find(v, 2); !it; ) { break; }
  // CHECK-MESSAGES: :[[@LINE-1]]:44: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
  // CHECK-FIXES: for (auto it = std::ranges::find(v, 2); (it == std::ranges::end(v)); ) { break; }
}

void test_invalid_fixit() {
  int arr[] = {1, 2, 3};
  if (int* it2 = std::find(arr, arr + 3, 2)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of standard algorithm used in boolean context; did you mean to compare with the end iterator? [bugprone-missing-end-comparison]
}
