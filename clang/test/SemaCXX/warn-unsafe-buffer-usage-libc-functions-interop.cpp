// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -Wno-error=bounds-safety-strict-terminated-by-cast\
// RUN:            -verify -fexperimental-bounds-safety-attributes %s
#include <ptrcheck.h>
#include <stddef.h>

typedef struct {} FILE;
typedef struct {} va_list;

namespace std {
  template <typename T>
  struct span {
    T *data() const noexcept;
    size_t size() const noexcept;
    size_t size_bytes() const noexcept;
    span<T> first(size_t count) const noexcept;
    span<T> last(size_t count) const noexcept;
    span<T> subspan(size_t offset, size_t count) const noexcept;
    const T &operator[](const size_t idx) const;
  };

  template <typename CharT>
  struct basic_string {
    const CharT *data() const noexcept;
    CharT *data() noexcept;
    const CharT *c_str() const noexcept;
    size_t size() const noexcept;
    size_t length() const noexcept;
  };

  typedef basic_string<char> string;
  typedef basic_string<wchar_t> wstring;
}

// The -Wunsafe-buffer-usage analysis considers some printf
// functions safe, arguments are correctly annotated. Because these
// functions are harder to be changed to C++ equivalents.
int printf(const char*, ... );
int fprintf (FILE*, const char*, ... );
int snprintf( char*, size_t, const char*, ... );
int snwprintf( wchar_t*, size_t, const wchar_t*, ... );
int vsnprintf( char*, size_t, const char*, va_list va_args );
// It is convenient to accept functions like `strlen` or `atoi` when
// they take a __null_termianted argument.
size_t strlen( const char* );
int atoi( const char* );

void test(const char * __null_terminated safe_str,
	  char * __sized_by(n) safe_p,
	  size_t n) {
  FILE *file;
  printf("%s", safe_str);
  fprintf(file, "%s", safe_str);
  snprintf(safe_p, n, "%s", safe_str);
  strlen(safe_str);
  atoi(safe_str);
  printf(safe_str);
  strlen(safe_p); // safe_p is not null-terminated  expected-warning{{function 'strlen' is unsafe}}

  // v-printf functions and sprintf are still warned about because
  // they cannot be fully safe:
  va_list vlist;
  vsnprintf(safe_p, n, "%s", vlist); // expected-warning{{function 'vsnprintf' is unsafe}} expected-note{{'va_list' is unsafe}}
}

void test_wchar(const wchar_t * unsafe_wstr,
		const wchar_t * __null_terminated safe_wstr,
	        wchar_t * __null_terminated nt_wstr,
		wchar_t * __sized_by(n) sizedby_wp, // a 'wchar_t' is larger than a `char`
		wchar_t * __counted_by(n) safe_wp,
		size_t n) {
  std::wstring cxx_wstr;
  std::span<wchar_t> cxx_wspan;

  snwprintf(cxx_wstr.data(), cxx_wstr.size(), cxx_wstr.c_str());
  snwprintf(safe_wp, n, safe_wstr);
  snwprintf(cxx_wspan.data(), cxx_wspan.size(), cxx_wspan.data()); // expected-warning{{function 'snwprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  snwprintf(sizedby_wp, n, safe_wstr); // expected-warning{{function 'snwprintf' is unsafe}} expected-note{{buffer pointer and size may not match}}
  snwprintf(safe_wp, n, unsafe_wstr); // expected-warning{{function 'snwprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
}

