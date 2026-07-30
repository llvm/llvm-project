// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

/* GCC was generating PHI nodes with an arity < #pred of the basic block the
 * PHI node lived in.  This was breaking LLVM because the number of entries
 * in a PHI node must equal the number of predecessors for a basic block.
 */

int trys(char *s, int x)
{
  int asa;
  double Val;
  int LLS;
  if (x) {
    asa = LLS + asa;
  } else {
  }
  return asa+(int)Val;
}

