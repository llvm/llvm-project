// RUN: %check_clang_tidy %s bugprone-signed-char-misuse %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-signed-char-misuse.DiagnoseSignedUnsignedCharComparisons: false \
// RUN:   }}"

int SignedToIntConversion() {
  signed char CCharacter = -5;
  int NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]
  return NCharacter;
}

int SignedUnsignedCharComparison(signed char SCharacter) {
  unsigned char USCharacter = 'a';
  if (SCharacter == USCharacter)
    return 1;
  return 0;
}
