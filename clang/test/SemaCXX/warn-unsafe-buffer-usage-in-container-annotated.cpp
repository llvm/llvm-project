// clang-format off
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage-in-container -verify %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -Wno-unsafe-buffer-usage-in-container -verify=nowarn %s

typedef unsigned long size_t;

namespace custom {
template <typename T>
class MyVector {
public:
  T* data() noexcept;
  size_t size() const noexcept;
  T* begin() noexcept;
  T* end() noexcept;
};
} // namespace custom

template <typename T>
class CustomSpan {
public:
  [[clang::unsafe_buffer_usage("container")]]
  CustomSpan(T* ptr, size_t size);

  template <typename It>
  [[clang::unsafe_buffer_usage("container")]]
  CustomSpan(It first, It last);
};

template <typename T>
[[clang::unsafe_buffer_usage("container")]]
CustomSpan<T> MakeCustomSpan(T* ptr, size_t size) {
  return CustomSpan<T>(ptr, size);
}

template <typename T>
[[clang::unsafe_buffer_usage("container")]]
CustomSpan<T> MakeCustomSpan(T* first, T* last) {
  return CustomSpan<T>(first, last);
}

void test_constructor(int* p, size_t n, custom::MyVector<int>& vec,
                      custom::MyVector<int>& vec2) {
  // Unsafe: decoupled pointer and size
  CustomSpan<int> s1(p, n); // expected-warning{{the two-parameter CustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  CustomSpan<int> s2(p, 10); // expected-warning{{the two-parameter CustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

  // Unsafe: duck typing approach with .data() and .size() called on different custom container objects
  CustomSpan<int> s_bad_data_size(vec.data(), vec2.size()); // expected-warning{{the two-parameter CustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

  // Unsafe: mismatched or reversed .begin() and .end()
  CustomSpan<int> s_bad_iter1(vec.begin(), vec2.end()); // expected-warning{{the two-parameter CustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  CustomSpan<int> s_bad_iter2(vec.end(), vec.begin()); // expected-warning{{the two-parameter CustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

  // Safe: duck typing approach with .data() and .size() called on the same custom container object
  CustomSpan<int> s3(vec.data(), vec.size()); // no-warning

  // Safe: .begin() and .end() called on the same custom container object
  CustomSpan<int> s4(vec.begin(), vec.end()); // no-warning

  // Safe: single element address
  int x;
  CustomSpan<int> s5(&x, 1); // no-warning

  // Safe: zero size
  CustomSpan<int> s6(p, 0); // no-warning

  // Safe: constant array
  int arr[10];
  CustomSpan<int> s7(arr, 10); // no-warning

  // Safe: new expression
  CustomSpan<int> s8(new int[10], 10); // no-warning
  CustomSpan<int> s9(new int[n], n); // no-warning
}

void test_factory(int* p, size_t n, custom::MyVector<int>& vec,
                  custom::MyVector<int>& vec2) {
  // Unsafe: decoupled pointer and size
  auto s1 = MakeCustomSpan(p, n); // expected-warning{{the two-parameter MakeCustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  auto s2 = MakeCustomSpan(p, 10); // expected-warning{{the two-parameter MakeCustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

  // Unsafe: duck typing approach with .data() and .size() called on different custom container objects
  auto s_bad_data_size = MakeCustomSpan(vec.data(), vec2.size()); // expected-warning{{the two-parameter MakeCustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

  // Unsafe: mismatched or reversed .begin() and .end()
  auto s_bad_iter1 = MakeCustomSpan(vec.begin(), vec2.end()); // expected-warning{{the two-parameter MakeCustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  auto s_bad_iter2 = MakeCustomSpan(vec.end(), vec.begin()); // expected-warning{{the two-parameter MakeCustomSpan construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

  // Safe: duck typing approach with .data() and .size() called on the same custom container object
  auto s3 = MakeCustomSpan(vec.data(), vec.size()); // no-warning

  // Safe: .begin() and .end() called on the same custom container object
  auto s4 = MakeCustomSpan(vec.begin(), vec.end()); // no-warning

  // Safe: single element address
  int x;
  auto s5 = MakeCustomSpan(&x, 1); // no-warning

  // Safe: zero size
  auto s6 = MakeCustomSpan(p, 0); // no-warning

  // Safe: constant array
  int arr[10];
  auto s7 = MakeCustomSpan(arr, 10); // no-warning
}

void test_pragma_suppression(int* p, size_t n) {
#pragma clang unsafe_buffer_usage begin
  CustomSpan<int> s1(p, n); // no-warning
  auto s2 = MakeCustomSpan(p, n); // no-warning
#pragma clang unsafe_buffer_usage end
}

// Test the [[clang::unsafe_buffer_usage_in_container]] spelling.
template <typename T>
class CustomSpanAlt {
public:
  [[clang::unsafe_buffer_usage_in_container]]
  CustomSpanAlt(T* ptr, size_t size);

  template <typename It>
  [[clang::unsafe_buffer_usage_in_container]]
  CustomSpanAlt(It first, It last);
};

template <typename T>
[[clang::unsafe_buffer_usage_in_container]]
CustomSpanAlt<T> MakeCustomSpanAlt(T* ptr, size_t size) {
  return CustomSpanAlt<T>(ptr, size);
}

void test_in_container_spelling(int* p, size_t n, custom::MyVector<int>& vec) {
  // Unsafe: decoupled pointer and size
  CustomSpanAlt<int> s1(p, n); // expected-warning{{the two-parameter CustomSpanAlt construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  auto s2 = MakeCustomSpanAlt(p, n); // expected-warning{{the two-parameter MakeCustomSpanAlt construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

  // Safe: duck typing approach with .data() and .size() called on the same custom container object
  CustomSpanAlt<int> s3(vec.data(), vec.size()); // no-warning
  auto s4 = MakeCustomSpanAlt(vec.data(), vec.size()); // no-warning
}

[[clang::unsafe_buffer_usage("invalid")]] // expected-warning{{'clang::unsafe_buffer_usage' attribute argument not supported: invalid}} nowarn-warning{{'clang::unsafe_buffer_usage' attribute argument not supported: invalid}}
void test_unsupported_category(int* p, size_t n);
