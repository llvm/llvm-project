// [C23] missing underlying types
enum E1 : int {
  E1Enumerator1
};

enum E2 {
  E2Enumerator1
};

// [C23] Incompatible underlying types
enum E3 : long {
  E3Enumerator1
};

