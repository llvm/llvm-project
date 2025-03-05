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

namespace annotated_libc {
  // For libc functions that have annotations,
  // `-Wunsafe-buffer-usage-in-libc-call` yields to the interoperation
  // warnings.

// expected-note@+2{{consider using a safe container and passing '.data()' to the parameter 'dst' and '.size()' to its dependent parameter 'size' or 'std::span' and passing '.first(...).data()' to the parameter 'dst'}}
// expected-note@+1{{consider using a safe container and passing '.data()' to the parameter 'src' and '.size()' to its dependent parameter 'size' or 'std::span' and passing '.first(...).data()' to the parameter 'src'}}
void *memcpy(void * __sized_by(size) dst, const void * __sized_by(size) src, size_t size);
size_t strlen( const char* __null_terminated str );
// expected-note@+1{{consider using a safe container and passing '.data()' to the parameter 'buffer' and '.size()' to its dependent parameter 'buf_size' or 'std::span' and passing '.first(...).data()' to the parameter 'buffer'}}
int snprintf( char* __counted_by(buf_size) buffer, size_t buf_size, const char* format, ... );
// expected-note@+1 2{{consider using a safe container and passing '.data()' to the parameter 'buffer' and '.size()' to its dependent parameter 'buf_size' or 'std::span' and passing '.first(...).data()' to the parameter 'buffer'}}
int snwprintf( wchar_t* __counted_by(buf_size) buffer, size_t buf_size, const wchar_t* format, ... );
int vsnprintf( char* __counted_by(buf_size) buffer, size_t buf_size, const char* format, va_list va_args);

// The '__counted_by(10)' is not a correct bounds annotation for
// 'sprintf'. It is used to test that even if 'sprintf' has bounds
// annotations, the function will still be warned against as 'sprintf'
// can't be safe.
int sprintf( char* __counted_by(10) buffer, const char* format, ... );

void test(char * p, char * q, const char * str,
	  const char * __null_terminated safe_str,
	  char * __counted_by(n) safe_p,
	  size_t n,
	  char * __counted_by(10) safe_ten) {
  memcpy(p, q, 10);                  // expected-warning2{{unsafe assignment to function parameter of count-attributed type}}
  snprintf(p, 10, "%s", "hlo");      // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  // We still warn about unsafe string pointer arguments to printfs:
  snprintf(safe_p, n, "%s", str);  // expected-warning{{function 'snprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}

  memcpy(safe_p, safe_p, n);               // no warn
  strlen(str);                             // expected-warning{{passing 'const char *' to parameter of incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation}}
  snprintf(safe_p, n, "%s", "hlo");        // no warn
  snprintf(safe_p, n, "%s", safe_str);     // no warn

  // v-printf functions and sprintf are still warned about because
  // they cannot be fully safe:
  va_list vlist;
  vsnprintf(safe_p, n, "%s", vlist); // expected-warning{{function 'vsnprintf' is unsafe}} expected-note{{'va_list' is unsafe}}
  sprintf(safe_ten, "%s", safe_str);    // expected-warning{{function 'sprintf' is unsafe}} expected-note{{change to 'snprintf' for explicit bounds checking}}

}

void test_wchar(wchar_t * p, wchar_t * q, const wchar_t * wstr,
	  const wchar_t * __null_terminated safe_wstr,
	  wchar_t * __null_terminated nt_wstr,
	  wchar_t * __counted_by(n) safe_p,
	  wchar_t * __sized_by(n) sizedby_p,
	  size_t n) {
  std::wstring cxx_wstr;
  std::span<wchar_t> cxx_wspan;

  snwprintf(safe_p, n, L"%ls", safe_wstr);
  snwprintf(cxx_wstr.data(), cxx_wstr.size(), cxx_wstr.c_str());
  snwprintf(cxx_wspan.data(), cxx_wspan.size(), cxx_wspan.data()); // expected-warning{{function 'snwprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
  snwprintf(p, n, L"%ls", safe_wstr); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  snwprintf(sizedby_p, n, L"%ls", safe_wstr); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  snwprintf(safe_p, n, L"%ls", wstr); // expected-warning{{function 'snwprintf' is unsafe}} expected-note{{string argument is not guaranteed to be null-terminated}}
}
} // namespace annotated_libc

namespace unannotated_libc {
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
} // namespace unannotated_libc
