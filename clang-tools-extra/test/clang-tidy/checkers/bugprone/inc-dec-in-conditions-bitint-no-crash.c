// RUN: %check_clang_tidy %s bugprone-inc-dec-in-conditions %t

_BitInt(8) v_401_0() {
  0 && ({
    _BitInt(5) y = 0;
    16777215wb ?: ++y;
  });
}
// CHECK-MESSAGES: warning 
