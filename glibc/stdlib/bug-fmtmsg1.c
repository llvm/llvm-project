#include <fmtmsg.h>
#include <stdio.h>


static int
do_test (void)
{
  /* Ugly, but fmtmsg would otherwise print to stderr which we do not
     want.  */
  fclose (stderr);
  stderr = stdout;

  int e1;
  e1 = fmtmsg (MM_PRINT, "label:part", MM_WARNING, "text", "action", "tag");

  int e2;
  e2 = fmtmsg (MM_PRINT, "label2:part2", 11, "text2", "action2", "tag2");

  addseverity (10, "additional severity");

  int e3;
  e3 = fmtmsg (MM_PRINT, "label3:part3", 10, "text3", "action3", "tag3");

  return e1 != 0 || e2 != 0 || e3 != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
