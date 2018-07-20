// RUN: %clang_cc1 -fchar8_t -std=c++2a -verify %s

char8_t a = u8'a';
char8_t b[] = u8"foo";
char8_t c = 'a';
char8_t d[] = "foo"; // expected-error {{initializing 'char8_t' array with plain string literal}} expected-note {{add 'u8' prefix}}

char e = u8'a';
char f[] = u8"foo"; // expected-error {{initialization of char array with UTF-8 string literal is not permitted by '-fchar8_t'}}
char g = 'a';
char h[] = "foo";

void disambig() {
  char8_t (a) = u8'x';
}

void operator""_a(char);
void operator""_a(const char*, decltype(sizeof(0)));

void test_udl1() {
  int &x = u8'a'_a; // expected-error {{no matching literal operator}}
  float &y = u8"a"_a; // expected-error {{no matching literal operator}}
}

int &operator""_a(char8_t);
float &operator""_a(const char8_t*, decltype(sizeof(0)));

void test_udl2() {
  int &x = u8'a'_a;
  float &y = u8"a"_a;
}

template<typename E, typename T> void check(T &&t) {
  using Check = E;
  using Check = T;
}
void check_deduction() {
  check<char8_t>(u8'a');
  check<const char8_t(&)[5]>(u8"a\u1000");
}

static_assert(sizeof(char8_t) == 1);
static_assert(char8_t(-1) > 0);
static_assert(u8"\u0080"[0] > 0);
