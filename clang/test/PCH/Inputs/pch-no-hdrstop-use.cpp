#include "Inputs/pch-through1.h"
static int bar() { return 42; }
#include "Inputs/pch-through2.h"
int pch();
#pragma hdrstop

//expected-no-diagnostics
int main()
{
  return pch() + through1(0) + through2(-1) + bar() - 42;
}
