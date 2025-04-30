/*
  WCSCHR: wchar_t *wcschr (wchar_t *ws, wchar_t wc);
*/

#define TST_FUNCTION wcschr

#include "tsp_common.c"
#include "dat_wcschr.c"

int
tst_wcschr (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  wchar_t *ws, wc;

  TST_DO_TEST (wcschr)
  {
    TST_HEAD_LOCALE (wcschr, S_WCSCHR);
    TST_DO_REC (wcschr)
    {
      TST_GET_ERRET (wcschr);
      ws = TST_INPUT (wcschr).ws;	/* external value: size WCSSIZE */
      wc = TST_INPUT (wcschr).wc;
      ret = wcschr (ws, wc);

      if (debug_flg)
	{
	  if (ret)
	    {
	      fprintf (stderr, "wcschr: ret = 0x%lx\n",
		       (unsigned long int) *ret);
	    }
	  else
	    {
	      fprintf (stderr, "wcschr: ret = NULL pointer\n");
	    }
	}

      TST_IF_RETURN (S_WCSCHR)
      {
	if (ret == NULL)
	  {
	    if (debug_flg)
	      {
		fprintf (stderr, "*** Warning *** tst_wcschr: "
			 "set ret_flg=1 to check NULL return value\n");
	      }

	    warn_count++;
	    Result (C_INVALID, S_WCSCHR, CASE_3, "(check the test data) "
		    "set ret_flg=1 to check NULL return value");
	    continue;
	  }

	if (*ret == wc)
	  {
	    Result (C_SUCCESS, S_WCSCHR, CASE_3, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_WCSCHR, CASE_3,
		    "the returned address of the string seems to be wrong");
	  }
      }
    }
  }

  return err_count;
}
