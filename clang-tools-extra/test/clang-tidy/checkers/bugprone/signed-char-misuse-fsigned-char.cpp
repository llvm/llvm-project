// RUN: %check_clang_tidy %s bugprone-signed-char-misuse %t -- -- -fsigned-char

int PlainChar() {
  char CCharacter = -5;
  int NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}
