/*
  WCTRANS: wctrans_t wctrans (const char *charclass);
*/

#define TST_FUNCTION wctrans

#include "tsp_common.c"
#include "dat_wctrans.c"

int
tst_wctrans (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wctrans_t);
  char *class;

  TST_DO_TEST (wctrans)
  {
    TST_HEAD_LOCALE (wctrans, S_WCTRANS);
    TST_DO_REC (wctrans)
    {
      TST_GET_ERRET (wctrans);
      class = TST_INPUT (wctrans).class;

      TST_CLEAR_ERRNO;
      ret = wctrans (class);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stderr, "tst_wctrans : [ %d ] ret = %ld\n", rec + 1,
		   (long int) ret);
	  fprintf (stderr, "		       errno = %d\n", errno_save);
	}

      TST_IF_RETURN (S_WCTRANS)
      {
	if (ret != 0)
	  {
	    Result (C_SUCCESS, S_WCTYPE, CASE_3, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_WCTYPE, CASE_3,
		    "should return non-0, but returned 0");
	  }
      }
    }
  }

  return err_count;
}
