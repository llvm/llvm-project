// RUN: %check_clang_tidy %s hicpp-avoid-goto %t

void f() {
  goto Exit;
  // CHECK-MESSAGES: warning: 'hicpp-avoid-goto' check is deprecated and will be removed in a future release; consider using 'cppcoreguidelines-avoid-goto' instead [clang-tidy-config]
  // CHECK-MESSAGES: [[@LINE-2]]:3: warning: avoid using 'goto' for flow control [hicpp-avoid-goto]
Exit:;
  // CHECK-MESSAGES: [[@LINE-1]]:1: note: label defined here
}
