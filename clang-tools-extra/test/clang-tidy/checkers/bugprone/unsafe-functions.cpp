// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-unsafe-functions %t

#include <utility>
#include <cstddef>

namespace std {

template<class T>
std::pair<T*, std::ptrdiff_t>
    get_temporary_buffer(std::ptrdiff_t count) noexcept;
}

void test() {
  (void)std::get_temporary_buffer<int>(64);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: function 'get_temporary_buffer<int>' returns uninitialized memory without performance advantages, was deprecated in C++17 and removed in C++20; 'operator new[]' should be used instead
}
