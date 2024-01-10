// RUN: %check_clang_tidy %s misc-ReturnZero %t

int main() {
  return 0; // Should trigger a warning.
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'return 0;' at the end of main [misc-ReturnZero]
  // CHECK-FIXES: {{^}}int main() {{{$}}
}