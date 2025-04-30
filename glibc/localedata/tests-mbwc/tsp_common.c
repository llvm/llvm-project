/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *  Main driver
 */


#define TST_FUNCTION_CALL(func) _TST_FUNCTION_CALL(func)
#define _TST_FUNCTION_CALL(func) tst ##_## func

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <errno.h>
#include <signal.h>

#include "tst_types.h"
#include "tgn_locdef.h"


int
main (int argc, char *argv[])
{
  int ret;
  int debug;

  debug = argc > 1 ? atoi (argv[1]) : 0;

  if (debug)
    {
      fprintf (stdout, "\nTST_MBWC ===> %s ...\n", argv[0]);
    }
  ret = TST_FUNCTION_CALL (TST_FUNCTION) (stdout, debug);

  return (ret != 0);
}

#define	 MAX_RESULT_REC	 132
char result_rec[MAX_RESULT_REC];


int
result (FILE * fp, char res, const char *func, const char *loc, int rec_no,
	int seq_no, int case_no, const char *msg)
{
  if (fp == NULL
      || strlen (func) + strlen (loc) + strlen (msg) + 32 > MAX_RESULT_REC)
    {
      fprintf (stderr,
	       "Warning: result(): can't write the result: %s:%s:%d:%d:%s\n",
	       func, loc, rec_no, case_no, msg);
      return 0;
    }

  sprintf (result_rec, "%s:%s:%d:%d:%d:%c:%s\n", func, loc, rec_no, seq_no,
	   case_no, res, msg);

  if (fputs (result_rec, fp) == EOF)
    {
      return 0;
    }

  return 1;
}
