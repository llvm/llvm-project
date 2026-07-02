// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O3 -emit-obj -o /dev/null %s
// REQUIRES: x86-registered-target

// Regression test for a stack overflow in ScalarEvolution. Computing the range
// of an affine addrec needs the loop's backedge-taken count, whose computation
// can recurse back into range computation through loop-guard reasoning (the
// llvm.assume calls below). For this heavily-unrolled nested loop the mutual
// recursion chained across a large number of versioned loops and exhausted the
// stack inside the "Induction Variable Users" analysis. ScalarEvolution now
// bounds the depth of this recursion, so compilation must complete.

short d[3][3][3];

void a(int l, short b, char c, short e, short f, short g) {
#pragma clang loop unroll(enable)
  for (int h; h < 23LL; h += 1LL)
    for (short i = 0; i < 4 + 22; i += -3224943361791975759LL - 40623) {
      __builtin_assume(e - 17695 == 23);
      __builtin_assume((f ? c >= l : b) - 20846 == 4);
      for (short j = 0; j < e - 17695; j += b - 20846)
        for (short k = ((int)g < 0 ? (int)g : 0) + 9; k < 0; k += 3)
          d[k][h][h] = 0;
    }
}
