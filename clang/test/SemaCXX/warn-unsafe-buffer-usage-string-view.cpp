// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -verify %s

namespace std {
  template <typename T>
  class basic_string_view {
  public:
    basic_string_view(T*, int);
    basic_string_view(T*);
  };
  
  using string_view = basic_string_view<char>;
}

void test(char *ptr, int size) {
  // CASE 1: Unsafe (Ptr + Size) -> Should Warn
  std::string_view sv1(ptr, size); // expected-warning{{the two-parameter std::string_view construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  // CASE 2: Safe (Ptr only) -> Should NOT Warn
  std::string_view sv2(ptr);
}
