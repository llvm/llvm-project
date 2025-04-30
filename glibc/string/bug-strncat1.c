/* Test case by Joseph S. Myers <jsm28@cam.ac.uk>.  */
#undef __USE_STRING_INLINES
#define __USE_STRING_INLINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libc-diag.h>

char d[3] = "\0\1\2";

int
main (void)
{
  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (8, 0)
  /* GCC 8 warns about strncat truncating output; this is deliberately
     tested here.  */
  DIAG_IGNORE_NEEDS_COMMENT (8, "-Wstringop-truncation");
#endif
  strncat (d, "\5\6", 1);
  DIAG_POP_NEEDS_COMMENT;
  if (d[0] != '\5')
    {
      puts ("d[0] != '\\5'");
      exit (1);
    }
  if (d[1] != '\0')
    {
      puts ("d[1] != '\\0'");
      exit (1);
    }
  if (d[2] != '\2')
    {
      puts ("d[2] != '\\2'");
      exit (1);
    }

  return 0;
}
