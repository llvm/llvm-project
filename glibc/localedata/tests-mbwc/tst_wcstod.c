/*
  WCSTOD: double wcstod (wchar_t *np, const wchar_t **endp);
*/

#define TST_FUNCTION wcstod

#include "tsp_common.c"
#include "dat_wcstod.c"

int
tst_wcstod (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (double);
  wchar_t *np, *endp, fwc;
  double val;

  TST_DO_TEST (wcstod)
  {
    TST_HEAD_LOCALE (wcstod, S_WCSTOD);
    TST_DO_REC (wcstod)
    {
      TST_GET_ERRET (wcstod);
      np = TST_INPUT (wcstod).np;

      TST_CLEAR_ERRNO;
      ret = wcstod (np, &endp);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stdout, "wcstod() [ %s : %d ] ret  = %f\n", locale,
		   rec + 1, ret);
	  fprintf (stdout, "			  *endp = 0x%lx\n",
		   (unsigned long int) *endp);
	}

      TST_IF_RETURN (S_WCSTOD)
      {
	if (ret != 0)
	  {
	    val = ret - TST_EXPECT (wcstod).val;
	    if (TST_ABS (val) < TST_DBL_EPS)
	      {
		Result (C_SUCCESS, S_WCSTOD, CASE_3, MS_PASSED);
	      }
	    else
	      {
		err_count++;
		Result (C_FAILURE, S_WCSTOD, CASE_3, "return value is wrong");
	      }
	  }
      }

      fwc = TST_EXPECT (wcstod).fwc;

      if (fwc == *endp)
	{
	  Result (C_SUCCESS, S_WCSTOD, CASE_4, MS_PASSED);
	}
      else
	{
	  err_count++;
	  Result (C_FAILURE, S_WCSTOD, CASE_4, "a final wc is wrong.");
	}
    }
  }

  return err_count;
}
