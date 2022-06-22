// RUN: %check_clang_tidy %s cert-dcl58-cpp %t -- -- -std=c++1z -I %clang_tidy_headers

#include "system-header-simulation.h"

namespace A {
  namespace B {
    int b;
  }
}

namespace A {
  namespace B {
    int c;
  }
}

namespace posix {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: modification of 'posix' namespace can result in undefined behavior [cert-dcl58-cpp]
  namespace vmi {
  }
}

namespace std {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: modification of 'std' namespace can
  int stdInt;
}

namespace foobar {
  namespace std {
    int bar;
  }
}

namespace posix::a {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: modification of 'posix' namespace 
}

enum class MyError {
  ErrorA,
  ErrorB
};

namespace std {
template <>
struct is_error_code_enum<MyError> : std::true_type {};

template<>
void swap<MyError>(MyError &a, MyError &b);
}

enum class MyError2 {
  Error2A,
  Error2B
};

namespace std {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: modification of 'std' namespace 
template <>
struct is_error_code_enum<MyError2> : std::true_type {};

int foobar;
}

using namespace std;

int x;

