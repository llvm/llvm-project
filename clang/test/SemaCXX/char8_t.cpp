// RUN: %clang_cc1 -fchar8_t -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++2a -verify=expected %s
// RUN: %clang_cc1 -std=c++2a -verify=expected -fno-signed-char %s


char8_t a = u8'a';
char8_t b[] = u8"foo";
char8_t c = 'a';
char8_t d[] = "foo"; // expected-error {{initializing 'char8_t' array with plain string literal}} expected-note {{add 'u8' prefix}}

char e = u8'a';
char g = 'a';
char h[] = "foo";

unsigned char i[] = u8"foo";
unsigned char j[] = { u8"foo" };
char k[] = u8"foo";
char l[] = { u8"foo" };
signed char m[] = u8"foo"; // expected-error {{initialization of char array with UTF-8 string literal is not permitted}}
signed char n[] = { u8"foo" }; // expected-error {{cannot initialize an array element of type 'signed char' with an lvalue of type 'const char8_t[4]'}}

const unsigned char* uptr = u8"foo"; // expected-error {{cannot initialize}}
const signed char* sptr = u8"foo"; // expected-error {{cannot initialize}}
const char* ptr = u8"foo"; // expected-error {{cannot initialize}}

template <typename T>
void check_values() {
  constexpr T c[] = {0, static_cast<T>(0xFF), 0x42};
  constexpr T a[] = u8"\x00\xFF\x42";

  static_assert(a[0] == c[0]);
  static_assert(a[1] == c[1]);
  static_assert(a[2] == c[2]);
}

void call_check_values() {
  check_values<char>();
  check_values<unsigned char>();
}

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

namespace ambiguous {

struct A {
	char8_t s[10];
};
struct B {
  char s[10];
};

void f(A); // expected-note {{candidate}}
void f(B); // expected-note {{candidate}}

int test() {
  f({u8"foo"}); // expected-error {{call to 'f' is ambiguous}}
}

}
