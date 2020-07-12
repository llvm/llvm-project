#include "main.h"
#include "stop.h"

/* main entry into PROGRAM */
void _QQmain();

/* C main stub */
int main(int argc, const char *argv[], const char *envp[])
{
  RTNAME(ProgramStart)(argc, argv, envp);
  _QQmain();
  RTNAME(ProgramEndStatement)();
  return 0;
}
