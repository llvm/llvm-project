// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fexpansion-limit=32 -verify

void g(int);

template <__SIZE_TYPE__ size>
struct String {
  char data[size];

  template <__SIZE_TYPE__ n>
  constexpr String(const char (&str)[n]) { __builtin_memcpy(data, str, n); }

  constexpr const char* begin() const { return data; }
  constexpr const char* end() const { return data + size - 1; }
};

template <__SIZE_TYPE__ n>
String(const char (&str)[n]) -> String<n>;

template <typename T, __SIZE_TYPE__ size>
struct Array {
  T data[size]{};
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
};

void expansion_size() {
  static constexpr Array<int, 32> almost_too_big;
  template for (auto x : almost_too_big) g(x);
  template for (constexpr auto x : almost_too_big) g(x);

  static constexpr Array<int, 33> too_big;
  template for (auto x : too_big) g(x); // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
  template for (constexpr auto x : too_big) g(x); // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}

  static constexpr String big{"1234567890123456789012345678901234567890234567890"};
  template for (auto x : big) g(x); // expected-error {{expansion size 49 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
  template for (constexpr auto x : big) g(x); // expected-error {{expansion size 49 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}

  template for (auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32}) g(x);
  template for (constexpr auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32}) g(x);

  template for (auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33}) g(x);
  template for (constexpr auto x : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // expected-error {{expansion size 33 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33}) g(x);

  int huge[1'000'000'000];
  template for (auto x : huge) {} // expected-error {{expansion size 1000000000 exceeds maximum configured size 32}} expected-note {{use -fexpansion-limit=N to adjust this limit}}
}

void array_too_big() {
  int ok[32];
  int too_big[33];

  template for (auto x : ok) {}
  template for (auto x : too_big) {} // expected-error {{expansion size 33 exceeds maximum configured size 32}} \
                                        expected-note {{use -fexpansion-limit=N to adjust this limit}}
}
