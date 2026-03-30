// RUN: %check_clang_tidy -std=c++11-or-later -check-suffixes=DEFAULT %s bugprone-unsafe-functions %t
// RUN: %check_clang_tidy -std=c++11-or-later -check-suffixes=DEPRECATED %s bugprone-unsafe-functions %t -- \
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.ReportDeprecatedFunctions: true}}"

#include <utility>
#include <cstddef>

namespace std {

template<class T>
std::pair<T*, std::ptrdiff_t>
    get_temporary_buffer(std::ptrdiff_t count) noexcept;
}

extern "C" {
int bcmp(const void *, const void *, std::size_t);
void bcopy(const void *, void *, std::size_t);
void bzero(void *, std::size_t);
int getpw(int, char *);
int vfork();
}

void test() {
  (void)std::get_temporary_buffer<int>(64);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:9: warning: function 'get_temporary_buffer<int>' returns uninitialized memory without performance advantages, was deprecated in C++17 and removed in C++20; 'operator new[]' should be used instead
  // CHECK-MESSAGES-DEPRECATED: :[[@LINE-2]]:9: warning: function 'get_temporary_buffer<int>' returns uninitialized memory without performance advantages, was deprecated in C++17 and removed in C++20; 'operator new[]' should be used instead

  char Buf[32] = {};
  char Other[32] = {};

  getpw(0, Buf);
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: function 'getpw' is dangerous as it may overflow the provided buffer; 'getpwuid' should be used instead
  // CHECK-MESSAGES-DEPRECATED: :[[@LINE-2]]:3: warning: function 'getpw' is dangerous as it may overflow the provided buffer; 'getpwuid' should be used instead

  bcmp(Buf, Other, sizeof(Buf));
  // CHECK-MESSAGES-DEPRECATED: :[[@LINE-1]]:3: warning: function 'bcmp' is deprecated; 'memcmp' should be used instead

  bcopy(Buf, Other, sizeof(Buf));
  // CHECK-MESSAGES-DEPRECATED: :[[@LINE-1]]:3: warning: function 'bcopy' is deprecated; 'memmove' should be used instead

  bzero(Buf, sizeof(Buf));
  // CHECK-MESSAGES-DEPRECATED: :[[@LINE-1]]:3: warning: function 'bzero' is deprecated; 'memset' should be used instead

  vfork();
  // CHECK-MESSAGES-DEPRECATED: :[[@LINE-1]]:3: warning: function 'vfork' is deprecated; 'posix_spawn' should be used instead
}
