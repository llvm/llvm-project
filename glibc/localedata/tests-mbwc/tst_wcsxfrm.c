/*
  WCSXFRM: size_t wcsxfrm (wchar_t *ws1, const wchar_t *ws2, size_t n);
*/

#define TST_FUNCTION wcsxfrm

#include "tsp_common.c"
#include "dat_wcsxfrm.c"

int
tst_wcsxfrm (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  wchar_t *org1, *org2;
  wchar_t frm1[MBSSIZE], frm2[MBSSIZE];
  size_t n1, n2;
  int ret_coll, ret_cmp;

  TST_DO_TEST (wcsxfrm)
  {
    TST_HEAD_LOCALE (wcsxfrm, S_WCSXFRM);
    TST_DO_REC (wcsxfrm)
    {
      TST_GET_ERRET (wcsxfrm);
      org1 = TST_INPUT (wcsxfrm).org1;
      org2 = TST_INPUT (wcsxfrm).org2;
      n1 = TST_INPUT (wcsxfrm).n1;
      n2 = TST_INPUT (wcsxfrm).n2;
      if (n1 < 0 || sizeof (frm1) < n1 || sizeof (frm2) < n2)
	{
	  warn_count++;
	  Result (C_IGNORED, S_WCSXFRM, CASE_9,
		  "input data n1 or n2 is invalid");
	  continue;
	}

      /* an errno and a return value are checked
	 only for 2nd wcsxfrm() call.
	 A result of 1st call is used to compare
	 those 2 values by using wcscmp().
       */

      TST_CLEAR_ERRNO;
      ret = wcsxfrm (frm1, org1, n1);	/* First call */
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stdout, "tst_wcsxfrm() : REC = %d\n", rec + 1);
	  fprintf (stdout, "tst_wcsxfrm() : 1st ret = %zu\n", ret);
	}

      if (ret == -1 || ret >= n1 || errno_save != 0)
	{
	  warn_count++;
	  Result (C_INVALID, S_WCSXFRM, CASE_8,
		  "got an error in fist wcsxfrm() call");
	  continue;
	}

      TST_CLEAR_ERRNO;
      /* Second call */
      ret = wcsxfrm (((n2 == 0) ? NULL : frm2), org2, n2);
      TST_SAVE_ERRNO;

      TST_IF_RETURN (S_WCSXFRM)
      {
      };

      if (n2 == 0 || ret >= n2 || errno != 0)
	{
#if 0
	  warn_count++;
	  Result (C_IGNORED, S_WCSXFRM, CASE_7, "did not get a result");
#endif
	  continue;
	}

      if (debug_flg)
	{
	  fprintf (stdout, "tst_wcsxfrm() : 2nd ret = %zu\n", ret);
	}

      /* wcscoll() */
      TST_CLEAR_ERRNO;
      /* depends on wcscoll() ... not good though ... */
      ret_coll = wcscoll (org1, org2);
      TST_SAVE_ERRNO;

      if (errno != 0)		/* bugs * bugs may got correct results ... */
	{
	  warn_count++;
	  Result (C_INVALID, S_WCSXFRM, CASE_6,
		  "got an error in wcscoll() call");
	  continue;
	}
      /* wcscmp() */
      ret_cmp = wcscmp (frm1, frm2);

      if ((ret_coll == ret_cmp) || (ret_coll > 0 && ret_cmp > 0)
	  || (ret_coll < 0 && ret_cmp < 0))
	{
	  Result (C_SUCCESS, S_WCSXFRM, CASE_3,
		  MS_PASSED " (depends on wcscoll & wcscmp)");
	}
      else
	{
	  err_count++;
	  Result (C_FAILURE, S_WCSXFRM, CASE_3,
		  "results from wcscoll & wcscmp() do not match");
	}

      if (debug_flg)
	{
	  fprintf (stdout, "tst_wcsxfrm() : coll = %d <-> %d = cmp\n",
		   ret_coll, ret_cmp);
	}
    }
  }

  return err_count;
}
