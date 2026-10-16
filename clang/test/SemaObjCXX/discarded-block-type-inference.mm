// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fobjc-arc -fblocks %s

void  block_receiver(int (^)() );

int f1() {
  if constexpr (0)
    (block_receiver)(^{ return 2; });
  return 1;
}

int f2() {
  if constexpr (0)
    return (^{ return 2; })();
  return 1;
}
