// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-span-param %t

namespace std {
using size_t = decltype(sizeof(0));
template <typename T, typename Alloc = void>
class vector {
public:
  using size_type = size_t;
  using const_iterator = const T *;
  const T &operator[](size_type i) const;
  T &operator[](size_type i);
  const T &at(size_type i) const;
  size_type size() const;
  bool empty() const;
  const T *data() const;
  const T &front() const;
  const T &back() const;
  const_iterator begin() const;
  const_iterator end() const;
};

template <typename T>
class span {};
} // namespace std

// Positive: only uses operator[] and size().
void read_size_index(const std::vector<int> &v) {
  // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: parameter 'v' can be changed to 'std::span'; it is only used for read-only access [modernize-use-span-param]
  for (std::size_t i = 0; i < v.size(); ++i)
    (void)v[i];
}

// Positive: only uses data() and size().
void read_data(const std::vector<int> &v) {
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: parameter 'v' can be changed to 'std::span'; it is only used for read-only access [modernize-use-span-param]
  const int *p = v.data();
  (void)v.size();
}

// Positive: only uses empty().
void read_empty(const std::vector<int> &v) {
  // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: parameter 'v' can be changed to 'std::span'; it is only used for read-only access [modernize-use-span-param]
  if (v.empty())
    return;
}

// Positive: range-for loop.
void range_for(const std::vector<int> &v) {
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: parameter 'v' can be changed to 'std::span'; it is only used for read-only access [modernize-use-span-param]
  for (int x : v)
    (void)x;
}

// Positive: passed to function taking const vector&.
void consumer(const std::vector<int> &);
void pass_to_const_ref(const std::vector<int> &v) {
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: parameter 'v' can be changed to 'std::span'; it is only used for read-only access [modernize-use-span-param]
  consumer(v);
}

// Negative: non-const reference (can mutate).
void mutating(std::vector<int> &v) {
  v[0];
}

// Negative: virtual method (can't change signature).
struct Base {
  virtual void process(const std::vector<int> &v);
};

// Negative: template function.
template <typename T>
void templated(const std::vector<T> &v) {
  (void)v.size();
}

// Negative: no body (declaration only).
void no_body(const std::vector<int> &v);
