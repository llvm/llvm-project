// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

void foo()
{
  char *ap;
  ap[1] == '-' && ap[2] == 0;
}

