#include "../../runtime/io-api.h"
#include "../../runtime/main.h"
#include "../../runtime/stop.h"
#include <cstring>

using namespace Fortran::runtime::io;

int main(int argc, const char *argv[], const char *envp[]) {
  RTNAME(ProgramStart)(argc, argv, envp);
  auto *io{IONAME(BeginExternalListOutput)()};
  const char str[]{"Hello, world!"};
  IONAME(OutputAscii)(io, str, std::strlen(str));
  IONAME(OutputInteger64)(io, 678);
  IONAME(OutputReal64)(io, 0.0);
  IONAME(OutputReal64)(io, 2.0 / 3.0);
  IONAME(OutputReal64)(io, 1.0e99);
  IONAME(OutputReal64)(io, 1.0 / 0.0);
  IONAME(OutputReal64)(io, -1.0 / 0.0);
  IONAME(OutputReal64)(io, 0.0 / 0.0);
  IONAME(OutputComplex64)(io, 123.0, -234.0);
  IONAME(OutputLogical)(io, false);
  IONAME(OutputLogical)(io, true);
  IONAME(EndIoStatement)(io);
  RTNAME(ProgramEndStatement)();
  return 0;
}
