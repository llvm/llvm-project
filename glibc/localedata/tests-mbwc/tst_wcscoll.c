/*
  WCSCOLL: int wcscoll (const wchar_t *ws1, const wchar_t *ws2);
*/

#define TST_FUNCTION wcscoll

#include "tsp_common.c"
#include "dat_wcscoll.c"

int
tst_wcscoll (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t *ws1, *ws2;
  int cmp;

  TST_DO_TEST (wcscoll)
  {
    TST_HEAD_LOCALE (wcscoll, S_WCSCOLL);
    TST_DO_REC (wcscoll)
    {
      TST_GET_ERRET (wcscoll);
      ws1 = TST_INPUT (wcscoll).ws1;	/* external value: size WCSSIZE */
      ws2 = TST_INPUT (wcscoll).ws2;

      TST_CLEAR_ERRNO;
      ret = wcscoll (ws1, ws2);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stderr, "tst_wcscoll: ret = %d\n", ret);
	}

      cmp = TST_EXPECT (wcscoll).cmp_flg;
      TST_IF_RETURN (S_WCSCOLL)
      {
	if (cmp != 0)
	  {
	    if ((cmp == 1 && ret > 0) || (cmp == -1 && ret < 0))
	      {
		Result (C_SUCCESS, S_WCSCOLL, CASE_3, MS_PASSED);
	      }
	    else
	      {
		err_count++;
		if (cmp == 1)
		  {
		    if (ret == 0)
		      Result (C_FAILURE, S_WCSCOLL, CASE_3,
			      "the return value should be positive"
			      " but it's zero.");
		    else
		      Result (C_FAILURE, S_WCSCOLL, CASE_3,
			      "the return value should be positive"
			      " but it's negative.");
		  }
		else
		  {
		    if (ret == 0)
		      Result (C_FAILURE, S_WCSCOLL, CASE_3,
			      "the return value should be negative"
			      " but it's zero.");
		    else
		      Result (C_FAILURE, S_WCSCOLL, CASE_3,
			      "the return value should be negative"
			      " but it's positive.");
		  }
	      }
	  }
      }
    }
  }

  return err_count;
}
