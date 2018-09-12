#include "Inputs/pch-through1.h"
static int bar() { return 42; }
#include "Inputs/pch-through2.h"
int pch();
#pragma hdrstop

//expected-no-diagnostics
//CHECK-NOT: FunctionDecl{{.*}}other
//CHECK: FunctionDecl{{.*}}main
int main()
{
  return pch() - 42*42 + bar() - 42 + through1(0) + through2(33);
}
