/*
  STRFMON: size_t strfmon (char *buf, size_t nbyte, const char *fmt, ...)
*/

#define TST_FUNCTION strfmon

#include "tsp_common.c"
#include "dat_strfmon.c"
#include <monetary.h>

int
tst_strfmon (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  char buf[MONSIZE], *mon;
  size_t nbt;
  char *fmt;
  double val;

  TST_DO_TEST (strfmon)
  {
    TST_HEAD_LOCALE (strfmon, S_STRFMON);
    TST_DO_REC (strfmon)
    {
      TST_GET_ERRET (strfmon);
      nbt = TST_INPUT (strfmon).nbytes;
      fmt = TST_INPUT (strfmon).fmt;
      val = TST_INPUT (strfmon).val;
      memset (buf, 0, MONSIZE);
      if (nbt > MONSIZE)
	{
	  err_count++;
	  Result (C_FAILURE, S_STRFMON, CASE_3, "buffer too small in test");
	  continue;
	}

      TST_CLEAR_ERRNO;
      ret = strfmon (buf, nbt, fmt, val, val, val);
      TST_SAVE_ERRNO;

      if (debug_flg)		/* seems fprintf doesn't update the errno */
	{
	  fprintf (stdout, "strfmon() [ %s : %d ]\n", locale, rec + 1);
	  fprintf (stdout, "	  : err = %d | %s\n", errno_save,
		   strerror (errno));
	  fprintf (stdout, "	  : ret = %zd; \t fmt = |%s|\n", ret, fmt);
	  fprintf (stdout, "	  : buf = |%s|\n\n", buf);
	}

      TST_IF_RETURN (S_STRFMON)
      {
      };
      if (errno != 0 || ret == -1)
	{
	  continue;
	}

      mon = TST_EXPECT (strfmon).mon;

      if (!strcmp (buf, mon))
	{
	  Result (C_SUCCESS, S_STRFMON, CASE_3, MS_PASSED);
	}
      else
	{
	  err_count++;
	  Result (C_FAILURE, S_STRFMON, CASE_3, "the formatted string is "
		  "different from an expected result");
	}
    }
  }

  return err_count;
}
