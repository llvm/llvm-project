// RUN: %check_clang_tidy %s bugprone-signed-char-misuse %t \
// RUN: -config='{CheckOptions: \
// RUN:  {bugprone-signed-char-misuse.CharTypedefsToIgnore: "sal_Int8;int8_t"}}' \
// RUN: --

///////////////////////////////////////////////////////////////////
/// Test cases correctly caught by the check.

// Check that a simple test case is still caught.
int SimpleAssignment() {
  signed char CCharacter = -5;
  int NCharacter;
  NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

typedef signed char sal_Char;

int TypedefNotInIgnorableList() {
  sal_Char CCharacter = 'a';
  int NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int isalpha(int);

int CctypeFunctionArgumentTypedefNotInIgnorableList(sal_Char SCharacter) {
  return isalpha(SCharacter);
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: 'signed char' to 'int' conversion passed to a <cctype> function; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]
}

///////////////////////////////////////////////////////////////////
/// Test cases correctly ignored by the check.

typedef signed char sal_Int8;

int OneIgnorableTypedef() {
  sal_Int8 CCharacter = 'a';
  int NCharacter = CCharacter;

  return NCharacter;
}

typedef signed char int8_t;

int OtherIgnorableTypedef() {
  int8_t CCharacter = 'a';
  int NCharacter = CCharacter;

  return NCharacter;
}

int CctypeFunctionArgumentIgnorableTypedef(sal_Int8 SCharacter) {
  return isalpha(SCharacter);
}

///////////////////////////////////////////////////////////////////
/// Test cases which should be caught by the check.

namespace boost {

template <class T>
class optional {
  T *member;

public:
  optional(T value) {
    member = new T(value);
  }

  T operator*() { return *member; }
};

} // namespace boost

int DereferenceWithTypdef(boost::optional<sal_Int8> param) {
  int NCharacter = *param;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}
