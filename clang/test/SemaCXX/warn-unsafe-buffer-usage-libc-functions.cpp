// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -verify %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -verify %s -x objective-c++
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage-in-libc-call \
// RUN:            -verify %s
// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage-in-libc-call \
// RUN:            -verify %s -DTEST_STD_NS

typedef struct {} FILE;
typedef unsigned int size_t;

#ifdef TEST_STD_NS
namespace std {
#endif

void memcpy();
void __asan_memcpy();
void strcpy();
void strcpy_s();
void wcscpy_s();
unsigned strlen( const char* str );
int fprintf( FILE* stream, const char* format, ... );
int printf( const char* format, ... );
int sprintf( char* buffer, const char* format, ... );
int swprintf( char* buffer, const char* format, ... );
int snprintf( char* buffer, unsigned buf_size, const char* format, ... );
int snwprintf( char* buffer, unsigned buf_size, const char* format, ... );
int snwprintf_s( char* buffer, unsigned buf_size, const char* format, ... );
int vsnprintf( char* buffer, unsigned buf_size, const char* format, ... );
int sscanf_s(const char * buffer, const char * format, ...);
int sscanf(const char * buffer, const char * format, ... );
int wprintf(const wchar_t* format, ... );
int __asan_printf();

#ifdef TEST_STD_NS
} //namespace std
using namespace std;
#endif

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
    unsigned size();
  };

  typedef basic_string<char> string;
  typedef basic_string<wchar_t> wstring;

  template<typename T>
  struct basic_string_view {
    T *c_str() const noexcept;
    T *data()  const noexcept;
    unsigned size();
    const T* begin() const noexcept;
    const T* end() const noexcept;
  };

  typedef basic_string_view<char> string_view;
  typedef basic_string_view<wchar_t> wstring_view;
}

void f(char * p, char * q, std::span<char> s, std::span<char> s2) {
  typedef FILE * _Nullable aligned_file_ptr_t __attribute__((align_value(64)));
  typedef char * _Nullable aligned_char_ptr_t __attribute__((align_value(64)));
  aligned_file_ptr_t fp;
  aligned_char_ptr_t cp;

  memcpy();                   // expected-warning{{function 'memcpy' is unsafe}}
  __builtin_memcpy(p, q, 64); // expected-warning{{function '__builtin_memcpy' is unsafe}}
  __builtin___memcpy_chk(p, q, 8, 64);  // expected-warning{{function '__builtin___memcpy_chk' is unsafe}}
  __asan_memcpy();                      // expected-warning{{function '__asan_memcpy' is unsafe}}
  strcpy();                   // expected-warning{{function 'strcpy' is unsafe}}
  strcpy_s();                 // expected-warning{{function 'strcpy_s' is unsafe}}
  wcscpy_s();                 // expected-warning{{function 'wcscpy_s' is unsafe}}
#ifdef TEST_STD_NS
  std::strcpy();              // expected-warning{{function 'strcpy' is unsafe}}
  std::memcpy();              // expected-warning{{function 'memcpy' is unsafe}}
#endif

  /* Test printfs */
  fprintf((FILE*)p, "%s%d", p, *p);  // expected-warning{{function 'fprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  printf("%s%d", // expected-warning{{function 'printf' is unsafe}}
	 p,    // expected-note{{string argument is not guaranteed to be null-terminated}} note attached to the unsafe argument
	 *p);
  printf(cp, p, *p); // expected-warning{{function 'printf' is unsafe}} // expected-note{{string argument is not guaranteed to be null-terminated}}
  sprintf(q, "%s%d", "hello", *p); // expected-warning{{function 'sprintf' is unsafe}} expected-note{{change to 'snprintf' for explicit bounds checking}}
  swprintf(q, "%s%d", "hello", *p); // expected-warning{{function 'swprintf' is unsafe}} expected-note{{change to 'snprintf' for explicit bounds checking}}
  snprintf(q, 10, "%s%d", "hello", *p); // expected-warning{{function 'snprintf' is unsafe}} expected-note{{buffer pointer and size may not match}}
  snprintf(cp, 10, "%s%d", "hello", *p); // expected-warning{{function 'snprintf' is unsafe}} expected-note{{buffer pointer and size may not match}}
  snprintf(s.data(), s2.size(), "%s%d", "hello", *p); // expected-warning{{function 'snprintf' is unsafe}} expected-note{{buffer pointer and size may not match}}
  snwprintf(s.data(), s2.size(), "%s%d", "hello", *p); // expected-warning{{function 'snwprintf' is unsafe}} expected-note{{buffer pointer and size may not match}}
  snwprintf_s(                      // expected-warning{{function 'snwprintf_s' is unsafe}}
	      s.data(),             // expected-note{{buffer pointer and size may not match}} // note attached to the buffer
	      s2.size(),
	      "%s%d", "hello", *p);
  vsnprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // expected-warning{{function 'vsnprintf' is unsafe}} expected-note{{'va_list' is unsafe}}
  sscanf(p, "%s%d", "hello", *p);    // expected-warning{{function 'sscanf' is unsafe}}
  sscanf_s(p, "%s%d", "hello", *p);  // expected-warning{{function 'sscanf_s' is unsafe}}
  fprintf((FILE*)p, "%P%d%p%i hello world %32s", *p, *p, p, *p, p); // expected-warning{{function 'fprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  fprintf(fp, "%P%d%p%i hello world %32s", *p, *p, p, *p, p); // expected-warning{{function 'fprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  wprintf(L"hello %s", p); // expected-warning{{function 'wprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}


  char a[10], b[11];
  int c[10];
  std::wstring WS;

  snprintf(a, sizeof(b), "%s", __PRETTY_FUNCTION__);         // expected-warning{{function 'snprintf' is unsafe}} expected-note{{buffer pointer and size may not match}}
  snprintf((char*)c, sizeof(c), "%s", __PRETTY_FUNCTION__);  // expected-warning{{function 'snprintf' is unsafe}} expected-note{{buffer pointer and size may not match}}
  fprintf((FILE*)p, "%P%d%p%i hello world %32s", *p, *p, p, *p, "hello"); // no warn
  fprintf(fp, "%P%d%p%i hello world %32s", *p, *p, p, *p, "hello"); // no warn
  printf("%s%d", "hello", *p); // no warn
  snprintf(s.data(), s.size_bytes(), "%s%d", "hello", *p); // no warn
  snprintf(s.data(), s.size_bytes(), "%s%d", __PRETTY_FUNCTION__, *p); // no warn
  snwprintf(s.data(), s.size_bytes(), "%s%d", __PRETTY_FUNCTION__, *p); // no warn
  snwprintf_s(s.data(), s.size_bytes(), "%s%d", __PRETTY_FUNCTION__, *p); // no warn
  wprintf(L"hello %ls", L"world"); // no warn
  wprintf(L"hello %ls", WS.c_str()); // no warn
  strlen("hello");// no warn
  __asan_printf();// a printf but no argument, so no warn
}

void safe_examples(std::string s1, int *p) {
  snprintf(s1.data(), s1.size_bytes(), "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", p, s1.c_str()); // no warn
  snprintf(s1.data(), s1.size_bytes(), s1.c_str(), __PRETTY_FUNCTION__, *p, "hello", s1.c_str());      // no warn
  printf("%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", p, s1.c_str());              // no warn
  printf(s1.c_str(), __PRETTY_FUNCTION__, *p, "hello", s1.c_str());                   // no warn
  fprintf((FILE*)0, "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", p, s1.c_str());   // no warn
  fprintf((FILE*)0, s1.c_str(), __PRETTY_FUNCTION__, *p, "hello", s1.c_str());        // no warn

  char a[10];
  char c = 'c';

  snprintf(a, sizeof a, "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", s1.c_str());         // no warn
  snprintf(a, sizeof(decltype(a)), "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", s1.c_str());          // no warn
  snprintf(a, 10, "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", s1.c_str());                // no warn
  snprintf(&c, 1, "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", s1.c_str());                // no warn
  snprintf(nullptr, 0, "%s%d%s%p%s", __PRETTY_FUNCTION__, *p, "hello", s1.c_str());           // no warn
}

void test_sarg_precision(std::string Str, std::string_view Sv, std::wstring_view WSv,
			 std::span<char> SpC, std::span<int> SpI) {
  printf("%.*s");
  printf("%.*s", (int)Str.size(), Str.data());
  printf("%.*s", (int)Str.size_bytes(), Str.data());
  printf("%.*s", (int)Sv.size(), Sv.data());
  printf("%.*s", (int)SpC.size(), SpC.data());
  printf("%.*s", SpC.size(), SpC.data());
  printf("%.*ls", WSv.size(), WSv.data());
  printf("%.*s", SpC.data());  // no warn because `SpC.data()` is passed to the precision while the actually string pointer is not given

  printf("%.*s", SpI.size(), SpI.data()); // expected-warning {{function 'printf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  printf("%.*s", SpI.size(), SpC.data()); // expected-warning {{function 'printf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  printf("%.*s", WSv.size(), WSv.data()); // expected-warning {{function 'printf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}

  char a[10];
  int  b[10];

  printf("%.10s", a);
  printf("%.11s", a); // expected-warning {{function 'printf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  printf("%.10s", b); // expected-warning {{function 'printf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
}


void g(char *begin, char *end, char *p, std::span<char> s) {
  std::copy(begin, end, p); // no warn
  std::copy(s.begin(), s.end(), s.begin()); // no warn
}

// warning gets turned off
void ff(char * p, char * q, std::span<char> s, std::span<char> s2) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-libc-call"
  memcpy();
  __builtin_memcpy(p, q, 64);
  __builtin___memcpy_chk(p, q, 8, 64);
  __asan_memcpy();
  strcpy();
#ifdef TEST_STD_NS
  std::strcpy();
  std::memcpy();
#endif
  strcpy_s();
  wcscpy_s();
#pragma clang diagnostic pop
}



// functions not in global scope or std:: namespace are not libc
// functions regardless of their names:
struct StrBuff
{
  void strcpy();
  void strcpy(char* dst);
  void memcpy(void *dst, const void *src, size_t size);
};

namespace NS {
  void strcpy();
  void strcpy(char* dst);
  void memcpy(void *dst, const void *src, size_t size);
}

namespace std {
  // class methods even in std namespace cannot be libc functions:
  struct LibC
  {
    void strcpy();
    void strcpy(char* dst);
    void memcpy(void *dst, const void *src, size_t size);
  };
}

void test(StrBuff& str)
{
  char buff[64];
  str.strcpy();
  str.strcpy(buff);
  str.memcpy(buff, buff, 64);
  NS::strcpy();
  NS::strcpy(buff);
  NS::memcpy(buff, buff, 64);

  std::LibC LibC;

  LibC.strcpy();
  LibC.strcpy(buff);
  LibC.memcpy(buff, buff, 64);
}
