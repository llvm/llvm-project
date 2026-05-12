// [C23] missing underlying types
enum E1 {
  E1Enumerator1
};

enum E2 : int {
  E2Enumerator1
};

// [C23] Incompatible underlying types
enum E3 : short {
  E3Enumerator1
};

