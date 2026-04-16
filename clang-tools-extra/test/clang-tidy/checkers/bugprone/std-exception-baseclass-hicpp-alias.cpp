// RUN: %check_clang_tidy %s hicpp-exception-baseclass %t -- -- -fcxx-exceptions

namespace std {
class exception {};
} // namespace std

void f() {
  throw 1;
  // CHECK-MESSAGES: warning: 'hicpp-exception-baseclass' check is deprecated and will be removed in a future release; consider using 'bugprone-std-exception-baseclass' instead [clang-tidy-config]
  // CHECK-MESSAGES: [[@LINE-2]]:9: warning: throwing an exception whose type 'int' is not derived from 'std::exception' [hicpp-exception-baseclass]
}
