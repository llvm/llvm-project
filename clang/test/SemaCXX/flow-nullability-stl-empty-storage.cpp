// Tests the compiler's STL contract recognition without relying on vendor
// header annotations.
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++20 %s -verify

namespace std {
template <class T, unsigned long N> struct array {
  T *_Nullable data();
};

template <class T> struct span {
  span();
  T *_Nullable data();
};
} // namespace std

void zero_length_array_data_warns() {
  std::array<int, 0> values;
  *values.data() = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

void nonempty_array_data_is_nonnull() {
  std::array<int, 1> values;
  *values.data() = 1;
}

// Empty spans can return null, but span accessors are intentionally allowlisted
// because warning on every span produces excessive false positives.
void default_span_data_is_allowlisted() {
  std::span<int> values;
  *values.data() = 1;
}
