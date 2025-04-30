/*
  TOWCTRANS: wint_t towctrans (wint_t wc, wctrans_t desc);
*/

#define TST_FUNCTION towctrans

#include "tsp_common.c"
#include "dat_towctrans.c"


int
tst_towctrans (FILE *fp, int debug_flg)
{
  TST_DECL_VARS (wint_t);
  wint_t wc;
  const char *ts;
  wctrans_t wto;

  TST_DO_TEST (towctrans)
    {
      TST_HEAD_LOCALE (towctrans, S_TOWCTRANS);
      TST_DO_REC (towctrans)
	{
	  TST_GET_ERRET (towctrans);
	  wc = TST_INPUT (towctrans).wc;
	  ts = TST_INPUT (towctrans).ts;

	  wto = wctrans (ts);

	  TST_CLEAR_ERRNO;
	  ret = towctrans (wc, wto);
	  TST_SAVE_ERRNO;

	  if (debug_flg)
	    {
	      fprintf (stdout, "towctrans() [ %s : %d ] ret = 0x%x\n",
		       locale, rec+1, ret);
	      fprintf (stdout, "		      errno = %d\n",
		       errno_save);
	    }

	  TST_IF_RETURN (S_TOWCTRANS)
	    {
	      if (ret != 0)
		{
		  result (fp, C_SUCCESS, S_TOWCTRANS, locale, rec+1,
			  seq_num+1, 3, MS_PASSED);
		}
	      else
		{
		  err_count++;
		  result (fp, C_FAILURE, S_TOWCTRANS, locale, rec+1,
			  seq_num+1, 3,
			  "the function returned 0, but should be non-zero");
		}
	    }
	}
    }

  return err_count;
}
