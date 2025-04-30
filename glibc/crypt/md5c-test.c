#include <crypt.h>
#include <string.h>

int
main (int argc, char *argv[])
{
  const char salt[] = "$1$saltstring";
  char *cp;
  int result = 0;

  cp = crypt ("Hello world!", salt);

  /* MD5 is disabled in FIPS mode.  */
  if (cp)
    result |= strcmp ("$1$saltstri$YMyguxXMBpd2TEZ.vS/3q1", cp);

  return result;
}
