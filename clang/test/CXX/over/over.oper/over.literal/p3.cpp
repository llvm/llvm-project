// RUN: %clang_cc1 -std=c++11 %s -verify

using size_t = decltype(sizeof(int));

// Acceptable parameter declarations
char operator "" _a(const char *);
char operator "" _a(const char []);
char operator "" _a(unsigned long long);
char operator "" _a(long double);
char operator "" _a(char);
char operator "" _a(const volatile char);
char operator "" _a(wchar_t);
char operator "" _a(char16_t);
char operator "" _a(char32_t);
char operator "" _a(const char *, size_t);
char operator "" _a(const wchar_t *, size_t);
char operator "" _a(const char16_t *, size_t);
char operator "" _a(const char32_t *, size_t);
char operator "" _a(const char [32], size_t);

// Unacceptable parameter declarations
char operator "" _b(); // expected-error {{parameter}}
char operator "" _b(const wchar_t *); // expected-error {{parameter}}
char operator "" _b(long long); // expected-error {{parameter}}
char operator "" _b(double); // expected-error {{parameter}}
char operator "" _b(short); // expected-error {{parameter}}
char operator "" _a(char, int = 0); // expected-error {{parameter}}
char operator "" _b(unsigned short); // expected-error {{parameter}}
char operator "" _b(signed char); // expected-error {{parameter}}
char operator "" _b(unsigned char); // expected-error {{parameter}}
char operator "" _b(const short *, size_t); // expected-error {{parameter}}
char operator "" _b(const unsigned short *, size_t); // expected-error {{parameter}}
char operator "" _b(const signed char *, size_t); // expected-error {{parameter}}
char operator "" _b(const unsigned char *, size_t); // expected-error {{parameter}}
char operator "" _a(const volatile char *, size_t); // expected-error {{parameter}}
char operator "" _a(volatile wchar_t *, size_t); // expected-error {{parameter}}
char operator "" _a(char16_t *, size_t); // expected-error {{parameter}}
char operator "" _a(const char32_t *, size_t, bool = false); // expected-error {{parameter}}
char operator "" _a(const char *, signed long); // expected-error {{parameter}}
char operator "" _a(const char *, size_t = 0); // expected-error {{default argument}}
