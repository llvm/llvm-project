// RUN: %check_clang_tidy %s google-readability-namespace-comments %t -- -- -std=c++17

namespace n1::n2 {
namespace n3 {
  // So that namespace is not empty and has at least 10 lines.
  // 1
  // 2
  // 3
  // 3
  // 4
  // 5
  // 6
  // 7
  // 8
  void f();
}}
// CHECK-MESSAGES: :[[@LINE-1]]:2: warning: namespace 'n3' not terminated with
// CHECK-MESSAGES: :[[@LINE-14]]:11: note: namespace 'n3' starts here
// CHECK-MESSAGES: :[[@LINE-3]]:3: warning: namespace 'n1::n2' not terminated with a closing comment [google-readability-namespace-comments]
// CHECK-MESSAGES: :[[@LINE-17]]:11: note: namespace 'n1::n2' starts here
// CHECK-FIXES: }  // namespace n3
// CHECK-FIXES: }  // namespace n1::n2

