// RUN: %check_clang_tidy -std=c++17-or-later -expect-clang-tidy-error %s modernize-use-scoped-lock %t -- -- -isystem %clang_tidy_headers

#include <mutex>

void f() {
  std::lock_guard<std::mutex> dont_crash {some_nonexistant_variable};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::scoped_lock' instead of 'std::lock_guard' [modernize-use-scoped-lock]
  // CHECK-MESSAGES: :[[@LINE-2]]:43: error: use of undeclared identifier 'some_nonexistant_variable' [clang-diagnostic-error]
}
