/*
This test makes sure that flang's runtime does not depend on the C++ runtime
library. It tries to link this simple file against libFortranRuntime.a with
a C compiler.

REQUIRES: c-compiler

RUN: %cc -std=c90 %s -I%runtimeincludes %libruntime -o /dev/null
*/

#include "entry-names.h"

/*
Manually add declarations for the runtime functions that we want to make sure
we're testing. We can't include any headers directly since they likely contain
C++ code that would explode here.
*/
double RTNAME(CpuTime)();

int main() {
  double x = RTNAME(CpuTime)();
  return x;
}
