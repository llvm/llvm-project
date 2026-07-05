// RUN: %check_clang_tidy %s cert-str34-c %t

// Check whether alias is actually working.
int SimpleVarDeclaration() {
  signed char CCharacter = -5;
  int NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [cert-str34-c]

  return NCharacter;
}

// Check whether bugprone-signed-char-misuse.DiagnoseSignedUnsignedCharComparisons option is set correctly.
int SignedUnsignedCharEquality(signed char SCharacter) {
  unsigned char USCharacter = 'a';
  if (SCharacter == USCharacter) // no warning
    return 1;
  return 0;
}
