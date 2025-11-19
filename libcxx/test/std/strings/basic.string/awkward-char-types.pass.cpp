#include <stdint.h>
#include <string>
#include <algorithm>
#include <cassert>

#include "test_macros.h"

template<typename Char>
void test_string() {
  // Make a test string.
  std::basic_string<Char> s;
  LIBCPP_ASSERT(s.size() == 0);

  // Append enough chars to it that we must have switched over from a short
  // string stored internally to a long one pointing to a dynamic buffer,
  // causing a reallocation.
  unsigned n = sizeof(s) / sizeof(Char) + 1;
  for (unsigned i = 0; i < n; i++) {
    s.push_back(Char::from_integer(i));
    LIBCPP_ASSERT(s.size() == i + 1);
  }

  // Check that all the chars were correctly copied during the realloc.
  for (unsigned i = 0; i < n; i++) {
    LIBCPP_ASSERT(s[i] == Char::from_integer(i));
  }
}

template<typename Integer, size_t N>
struct TestChar {
  Integer values[N];

  static TestChar from_integer(unsigned index) {
    TestChar ch;
    for (size_t i = 0; i < N; i++)
      ch.values[i] = index + i;
    return ch;
  }

  bool operator==(const TestChar &other) const {
    return 0 == memcmp(values, other.values, sizeof(values));
  }
  bool operator<(const TestChar &other) const {
    return 0 < memcmp(values, other.values, sizeof(values));
  }
};

template<typename Integer, size_t N>
struct std::char_traits<TestChar<Integer, N>> {
  using char_type  = TestChar<Integer, N>;
  using int_type   = int;
  using off_type   = streamoff;
  using pos_type   = streampos;
  using state_type = mbstate_t;

  static TEST_CONSTEXPR_CXX20 void assign(char_type& c1, const char_type& c2) { c1 = c2; }
  static bool eq(char_type c1, char_type c2);
  static bool lt(char_type c1, char_type c2);

  static int compare(const char_type* s1, const char_type* s2, std::size_t n);
  static std::size_t length(const char_type* s);
  static const char_type* find(const char_type* s, std::size_t n, const char_type& a);
  static char_type* move(char_type* s1, const char_type* s2, std::size_t n);
  static TEST_CONSTEXPR_CXX20 char_type* copy(char_type* s1, const char_type* s2, std::size_t n) {
    std::copy_n(s2, n, s1);
    return s1;
  }
  static TEST_CONSTEXPR_CXX20 char_type* assign(char_type* s, std::size_t n, char_type a) {
    std::fill_n(s, n, a);
    return s;
  }

  static int_type not_eof(int_type c);
  static char_type to_char_type(int_type c);
  static int_type to_int_type(char_type c);
  static bool eq_int_type(int_type c1, int_type c2);
  static int_type eof();
};

int main(int, char**) {
  test_string<TestChar<uint8_t, 1>>();
  test_string<TestChar<uint8_t, 2>>();
  test_string<TestChar<uint8_t, 3>>();
  test_string<TestChar<uint8_t, 4>>();
  test_string<TestChar<uint8_t, 5>>();
  test_string<TestChar<uint8_t, 6>>();
  test_string<TestChar<uint8_t, 7>>();
  test_string<TestChar<uint8_t, 8>>();
  test_string<TestChar<uint8_t, 9>>();
  test_string<TestChar<uint8_t, 10>>();
  test_string<TestChar<uint8_t, 11>>();
  test_string<TestChar<uint8_t, 12>>();
  test_string<TestChar<uint8_t, 13>>();
  test_string<TestChar<uint8_t, 14>>();
  test_string<TestChar<uint8_t, 15>>();
  test_string<TestChar<uint8_t, 16>>();

  test_string<TestChar<uint16_t, 1>>();
  test_string<TestChar<uint16_t, 2>>();
  test_string<TestChar<uint16_t, 3>>();
  test_string<TestChar<uint16_t, 4>>();
  test_string<TestChar<uint16_t, 5>>();
  test_string<TestChar<uint16_t, 6>>();
  test_string<TestChar<uint16_t, 7>>();
  test_string<TestChar<uint16_t, 8>>();

  test_string<TestChar<uint32_t, 1>>();
  test_string<TestChar<uint32_t, 2>>();
  test_string<TestChar<uint32_t, 3>>();
  test_string<TestChar<uint32_t, 4>>();

  test_string<TestChar<uint64_t, 1>>();
  test_string<TestChar<uint64_t, 2>>();
}
