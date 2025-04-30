/*
  ISWCTYPE: int iswctype (wint_t wc, wctype_t desc);
*/

#define TST_FUNCTION iswctype

#include "tsp_common.c"
#include "dat_iswctype.c"


int
tst_iswctype (FILE *fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wint_t wc;
  const char *ts;

  TST_DO_TEST (iswctype)
    {
      TST_HEAD_LOCALE (iswctype, S_ISWCTYPE);
      TST_DO_REC (iswctype)
	{
	  TST_GET_ERRET (iswctype);
	  wc = TST_INPUT (iswctype).wc;
	  ts = TST_INPUT (iswctype).ts;
	  ret = iswctype (wc, wctype (ts));
	  TST_SAVE_ERRNO;
	  if (debug_flg)
	    {
	      fprintf (stdout, "iswctype() [ %s : %d ] ret = %d\n",
		       locale, rec+1, ret);
	    }

	  TST_IF_RETURN (S_ISWCTYPE)
	    {
	      if (ret != 0)
		{
		  result (fp, C_SUCCESS, S_ISWCTYPE, locale, rec+1,
			  seq_num+1, 3, MS_PASSED);
		}
	      else
		{
		  err_count++;
		  result (fp, C_FAILURE, S_ISWCTYPE, locale, rec+1,
			  seq_num+1, 3,
			  "the function returned 0, but should be non-zero");
		}
	    }
	}
    }

  return err_count;
}
