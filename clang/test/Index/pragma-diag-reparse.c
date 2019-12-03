#pragma clang diagnostic ignored "-Wtautological-compare"
#include "pragma_disable_warning.h"

int main (int argc, const char * argv[])
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  int x=0;
#pragma clang diagnostic pop

  return x;
}

#pragma clang diagnostic ignored "-Wmisleading-indentation"
void foo() { int b=0; while (b==b); }

// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_FAILONERROR=1 c-index-test -test-load-source-reparse 5 local \
// RUN: -I%S/Inputs \
// RUN:   %s -Wall -Werror | FileCheck %s

// CHECK: pragma-diag-reparse.c:8:7: VarDecl=x:8:7 (Definition) Extent=[8:3 - 8:10]
