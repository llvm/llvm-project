// RUN: %check_clang_tidy %s bugprone-signed-char-misuse %t -- -- -funsigned-char

// Test framework needs something to catch, otherwise it fails.
int SignedChar() {
  signed char CCharacter = -5;
  int NCharacter;
  NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int PlainChar() {
  char CCharacter = -5;
  int NCharacter = CCharacter; // no warning
  return NCharacter;
}
