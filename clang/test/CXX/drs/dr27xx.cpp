// RUN: %clang_cc1 -std=c++2c -verify %s

namespace dr2798 { // dr2798: 17 drafting
#if __cpp_static_assert >= 202306
struct string {
  constexpr string() {
    data_ = new char[6]();
    __builtin_memcpy(data_, "Hello", 5);
    data_[5] = 0;
  }
  constexpr ~string() { delete[] data_; }
  constexpr unsigned long size() const { return 5; };
  constexpr const char *data() const { return data_; }

  char *data_;
};
struct X {
  string s;
};
consteval X f() { return {}; }

static_assert(false, f().s); // expected-error {{static assertion failed: Hello}}
#endif
} // namespace dr2798
