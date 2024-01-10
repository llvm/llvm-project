// RUN: %check_clang_tidy %s readability-DoNotReturnZeroCheck %t


int main() {
  return 0; // Should trigger a warning.
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'return 0;' at the end of main [readability-DoNotReturnZeroCheck]
  // CHECK-FIXES: {{^}}int main() {{{$}}
}