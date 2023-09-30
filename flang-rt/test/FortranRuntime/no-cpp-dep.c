/*
This test makes sure that flang's runtime does not depend on the C++ runtime
library. It tries to link this simple file against libFortranRuntime.a with
a C compiler.

REQUIRES: c-compiler

RUN: %cc -std=c99 %s -I%include %libruntime -lm -o /dev/null
*/

#include "flang/Runtime/entry-names.h"
#include <stdint.h>

/*
Manually add declarations for the runtime functions that we want to make sure
we're testing. We can't include any headers directly since they likely contain
C++ code that would explode here.
*/
struct EnvironmentDefaultList;
struct Descriptor;

double RTNAME(CpuTime)();

void RTNAME(ProgramStart)(
    int, const char *[], const char *[], const struct EnvironmentDefaultList *);
int32_t RTNAME(ArgumentCount)();
int32_t RTNAME(GetCommandArgument)(int32_t, const struct Descriptor *,
    const struct Descriptor *, const struct Descriptor *);
int32_t RTNAME(GetEnvVariable)();

int main() {
  double x = RTNAME(CpuTime)();
  RTNAME(ProgramStart)(0, 0, 0, 0);
  int32_t c = RTNAME(ArgumentCount)();
  int32_t v = RTNAME(GetCommandArgument)(0, 0, 0, 0);
  int32_t e = RTNAME(GetEnvVariable)("FOO", 0, 0);
  return x + c + v + e;
}
