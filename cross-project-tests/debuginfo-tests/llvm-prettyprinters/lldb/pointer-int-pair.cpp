#include <cstdio>

#include "llvm/ADT/PointerIntPair.h"

int main() {
  float a = 5;
  llvm::PointerIntPair<float *, 1, bool> float_pair(&a, true);
  llvm::PointerIntPair<void *, 1, bool> void_pair(&a, false);
  llvm::PointerIntPair<llvm::PointerIntPair<void *, 1, bool>, 1, bool> nested(
      void_pair, true);

  struct S {
    int i;
  };
  S s;

  enum class E : unsigned {
    Case1,
    Case2,
    Case3,
    Case4,
  };
  llvm::PointerIntPair<S *, 2, E> enum_pair(&s, E::Case2);

  S s2;

  puts("Break here");

  enum_pair.setPointerAndInt(&s2, E::Case3);

  puts("Break here");
}
