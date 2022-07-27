// RUN: %check_clang_tidy %s misc-misleading-identifier %t

int printf(const char *format, ...);

// CHECK-MESSAGES: :[[#@LINE+1]]:1: warning: identifier has right-to-left codepoints [misc-misleading-identifier]
short int א = (short int)0;
// CHECK-MESSAGES: :[[#@LINE+1]]:1: warning: identifier has right-to-left codepoints [misc-misleading-identifier]
short int ג = (short int)12345;

int main() {
  // CHECK-MESSAGES: :[[#@LINE+1]]:3: warning: identifier has right-to-left codepoints [misc-misleading-identifier]
  int א = ג; // a local variable, set to zero?
  printf("ג is %d\n", ג);
  printf("א is %d\n", א);
}
