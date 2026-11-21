// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify %s

template <typename T>
class CustomSpan {
  T *ptr_;

public:
  [[clang::unsafe_buffer_usage_in_data_call]]
  constexpr T *data() const noexcept { return ptr_; }

  [[clang::unsafe_buffer_usage("data")]]
  constexpr const T *raw_bytes() const noexcept { return ptr_; }

  __attribute__((__unsafe_buffer_usage_in_data_call__))
  constexpr T *gnu_data() const noexcept { return ptr_; }

  constexpr T *unannotated_data() const noexcept { return ptr_; }
};

struct Large {
  int a, b, c;
};

void test_annotated_methods(CustomSpan<int> s, CustomSpan<int> *sp) {
  // Unsafe casts on [[clang::unsafe_buffer_usage_in_data_call]]:
  Large *l1 = (Large *)s.data(); // expected-warning{{unsafe invocation of 'data'}}
  l1 = (Large *)sp->data(); // expected-warning{{unsafe invocation of 'data'}}

  // Unsafe casts on [[clang::unsafe_buffer_usage("data")]]:
  const Large *l2 = (const Large *)s.raw_bytes(); // expected-warning{{unsafe invocation of 'raw_bytes'}}

  // Unsafe casts on GNU spelling with double underscores:
  Large *l3 = (Large *)s.gnu_data(); // expected-warning{{unsafe invocation of 'gnu_data'}}

  // Unannotated methods do not warn:
  Large *l4 = (Large *)s.unannotated_data(); // no-warning
}
