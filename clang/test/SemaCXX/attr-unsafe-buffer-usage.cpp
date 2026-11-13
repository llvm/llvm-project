// RUN: %clang_cc1  -fsyntax-only -verify %s

// Function annotations.
[[clang::unsafe_buffer_usage]]
void f(int *buf, int size);
void g(int *buffer [[clang::unsafe_buffer_usage("buffer")]], int size); // expected-warning {{'clang::unsafe_buffer_usage' attribute only applies to functions}}

struct DataHolder {
  [[clang::unsafe_buffer_usage_in_data_call]]
  int *data();
  [[clang::unsafe_buffer_usage("data")]]
  int *data_str();
  [[clang::unsafe_buffer_usage_in_data_call("data")]] // expected-error {{'clang::unsafe_buffer_usage_in_data_call' attribute takes no arguments}}
  int *data_err();
  [[clang::unsafe_buffer_usage("invalid")]] // expected-warning {{'clang::unsafe_buffer_usage' attribute argument not supported: invalid}}
  int *invalid_category();
};
