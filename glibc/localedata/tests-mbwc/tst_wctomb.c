/*
  WCTOMB: int wctomb (char *s, wchar_t wc)
*/

#define TST_FUNCTION wctomb

#include "tsp_common.c"
#include "dat_wctomb.c"

int
tst_wctomb (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t wc;
  char s[MBSSIZE], *s_in, *s_ex;
  int err, i;

  TST_DO_TEST (wctomb)
  {
    TST_HEAD_LOCALE (wctomb, S_WCTOMB);
    TST_DO_REC (wctomb)
    {
      TST_GET_ERRET (wctomb);
      wc = TST_INPUT (wctomb).wc;
      s_in = ((TST_INPUT (wctomb).s_flg) == 0) ? (char *) NULL : s;
      ret = wctomb (s_in, wc);

      if (debug_flg)
	{
	  fprintf (stdout, "wctomb() [ %s : %d ] ret  = %d\n", locale,
		   rec + 1, ret);
	}

      TST_IF_RETURN (S_WCTOMB)
      {
	if (s_in == NULL)	/* state dependency */
	  {
	    if (ret_exp == +1)	/* state-dependent  */
	      {
		if (ret != 0)
		  {
		    /* Non-zero means state-dependent encoding.	 */
		    Result (C_SUCCESS, S_WCTOMB, CASE_3, MS_PASSED);
		  }
		else
		  {
		    err_count++;
		    Result (C_FAILURE, S_WCTOMB, CASE_3,
			    "should be state-dependent encoding, "
			    "but a return value shows it is "
			    "state-independent");
		  }
	      }

	    if (ret_exp == 0)	/* state-independent */
	      {
		if (ret == 0)
		  {
		    /* Non-zero means state-dependent encoding.	 */
		    Result (C_SUCCESS, S_WCTOMB, CASE_3, MS_PASSED);
		  }
		else
		  {
		    err_count++;
		    Result (C_FAILURE, S_WCTOMB, CASE_3,
			    "should be state-independent encoding, "
			    "but a return value shows it is state-dependent");
		  }
	      }
	  }
      }

      s_ex = TST_EXPECT (wctomb).s;

      if (s_in)
	{
	  for (i = 0, err = 0; *(s_ex + i) != 0 && i < MBSSIZE; i++)
	    {
	      if (s_in[i] != s_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCTOMB, CASE_4,
			  "copied string is different from an"
			  " expected string");
		  break;
		}
	    }

	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCTOMB, CASE_4, MS_PASSED);
	    }
	}
    }
  }

  return err_count;
}
