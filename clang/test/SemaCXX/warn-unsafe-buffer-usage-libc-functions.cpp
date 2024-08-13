// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -verify %s

typedef struct {} FILE;
void memcpy();
void __asan_memcpy();
void strcpy();
void strcpy_s();
void wcscpy_s();
unsigned strlen( const char* str );
int fprintf( FILE* stream, const char* format, ... );
int printf( const char* format, ... );
int sprintf( char* buffer, const char* format, ... );
int snprintf( char* buffer, unsigned buf_size, const char* format, ... );
int vsnprintf( char* buffer, unsigned buf_size, const char* format, ... );
int sscanf_s(const char * buffer, const char * format, ...);
int sscanf(const char * buffer, const char * format, ... );

namespace std {
  template< class InputIt, class OutputIt >
  OutputIt copy( InputIt first, InputIt last,
		 OutputIt d_first );

  struct iterator{};
  template<typename T>
  struct span {
    T * ptr;
    T * data();
    unsigned size_bytes();
    unsigned size();
    iterator begin() const noexcept;
    iterator end() const noexcept;
  };

  template<typename T>
  struct basic_string {
    T* p;
    T *c_str();
    T *data();
    unsigned size_bytes();
  };

  typedef basic_string<char> string;
  typedef basic_string<wchar_t> wstring;

  // C function under std:
  void memcpy();
  void strcpy();
}

void f(char * p, char * q, std::span<char> s, std::span<char> s2) {
  memcpy();                   // expected-warning{{function memcpy introduces unsafe buffer access}}
  std::memcpy();              // expected-warning{{function memcpy introduces unsafe buffer access}}
  __builtin_memcpy(p, q, 64); // expected-warning{{function __builtin_memcpy introduces unsafe buffer access}}
  __builtin___memcpy_chk(p, q, 8, 64);  // expected-warning{{function __builtin___memcpy_chk introduces unsafe buffer access}}
  __asan_memcpy();                      // expected-warning{{function __asan_memcpy introduces unsafe buffer access}}
  strcpy();                   // expected-warning{{function strcpy introduces unsafe buffer access}}
  std::strcpy();              // expected-warning{{function strcpy introduces unsafe buffer access}}
  strcpy_s();                 // expected-warning{{function strcpy_s introduces unsafe buffer access}}
  wcscpy_s();                 // expected-warning{{function wcscpy_s introduces unsafe buffer access}}


  /* Test printfs */
  fprintf((FILE*)p, "%s%d", p, *p);  // expected-warning{{function fprintf introduces unsafe buffer access}} expected-note{{use 'std::string::c_str' or string literal as string pointer to guarantee null-termination}}
  printf("%s%d", p, *p);  // expected-warning{{function printf introduces unsafe buffer access}} expected-note{{use 'std::string::c_str' or string literal as string pointer to guarantee null-termination}}
  sprintf(q, "%s%d", "hello", *p); // expected-warning{{function sprintf introduces unsafe buffer access}} expected-note{{change to 'snprintf' for explicit bounds checking}}
  snprintf(q, 10, "%s%d", "hello", *p); // expected-warning{{function snprintf introduces unsafe buffer access}} expected-note{{buffer pointer and size may not match}}
  snprintf(s.data(), s2.size(), "%s%d", "hello", *p); // expected-warning{{function snprintf introduces unsafe buffer access}} expected-note{{buffer pointer and size may not match}}
  vsnprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // expected-warning{{function vsnprintf introduces unsafe buffer access}} expected-note{{do not use va_list that cannot be checked at compile-time for bounds safety}}
  sscanf(p, "%s%d", "hello", *p);    // expected-warning{{function sscanf introduces unsafe buffer access}}
  sscanf_s(p, "%s%d", "hello", *p);  // expected-warning{{function sscanf_s introduces unsafe buffer access}}
  fprintf((FILE*)p, "%P%d%p%i hello world %32s", *p, *p, p, *p, p); // expected-warning{{function fprintf introduces unsafe buffer access}} expected-note{{use 'std::string::c_str' or string literal as string pointer to guarantee null-termination}}
  fprintf((FILE*)p, "%P%d%p%i hello world %32s", *p, *p, p, *p, "hello"); // no warn
  printf("%s%d", "hello", *p); // no warn
  snprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // no warn
  snprintf(s.data(), s.size_bytes(), "%s%d", __PRETTY_FUNCTION__, *p); // no warn
  strlen("hello");// no warn
}

void v(std::string s1, int *p) {
  snprintf(s1.data(), s1.size_bytes(), "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", p, s1.c_str()); // no warn
  snprintf(s1.data(), s1.size_bytes(), s1.c_str(), __PRETTY_FUNCTION__, *p, "hello", s1.c_str());      // no warn
  printf("%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", p, s1.c_str());              // no warn
  printf(s1.c_str(), __PRETTY_FUNCTION__, *p, "hello", s1.c_str());                   // no warn
  fprintf((FILE*)0, "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", p, s1.c_str());   // no warn
  fprintf((FILE*)0, s1.c_str(), __PRETTY_FUNCTION__, *p, "hello", s1.c_str());        // no warn
}


void g(char *begin, char *end, char *p, std::span<char> s) {
  std::copy(begin, end, p); // no warn
  std::copy(s.begin(), s.end(), s.begin()); // no warn
}
