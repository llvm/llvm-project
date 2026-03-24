// RUN: %check_clang_tidy %s cppcoreguidelines-pro-bounds-constant-array-index %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     cppcoreguidelines-pro-bounds-constant-array-index.GslHeader: 'dir1/gslheader.h', \
// RUN:     cppcoreguidelines-pro-bounds-constant-array-index.IncludeStyle: 'google' \
// RUN:   }}"

// CHECK-FIXES: #include "dir1/gslheader.h"

typedef __SIZE_TYPE__ size_t;

namespace std {
  template<typename T, size_t N>
  struct array {
    T& operator[](size_t n);
    T& at(size_t n);
  };
}

namespace gsl {
  template<class T, size_t N>
  T& at( std::array<T, N> &a, size_t index );
}

void f(std::array<int, 10> a, int pos) {
  a [ pos / 2 ] = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use array subscript when the index is not an integer constant expression [cppcoreguidelines-pro-bounds-constant-array-index]
  // CHECK-FIXES: gsl::at(a,  pos / 2 ) = 1;
}
