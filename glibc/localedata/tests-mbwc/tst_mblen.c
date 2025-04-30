/*
  MBLEN: int mblen (char *s, size_t n)
*/

#define TST_FUNCTION mblen

#include "tsp_common.c"
#include "dat_mblen.c"

int
tst_mblen (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  char s_flg;
  const char *s_in;
  size_t n;

  TST_DO_TEST (mblen)
  {
    TST_HEAD_LOCALE (mblen, S_MBLEN);
    TST_DO_REC (mblen)
    {
      TST_GET_ERRET (mblen);
      s_flg = TST_INPUT (mblen).s_flg;
      s_in = TST_INPUT (mblen).s;
      n = TST_INPUT (mblen).n;

      if (s_flg == 0)
	{
	  s_in = NULL;
	}

      if (n == USE_MBCURMAX)
	{
	  n = MB_CUR_MAX;
	}

      TST_CLEAR_ERRNO;
      ret = mblen (s_in, n);
      TST_SAVE_ERRNO;

      TST_IF_RETURN (S_MBLEN)
      {
	if (s_in == NULL)
	  {			/* state dependency */
	    if (ret_exp == +1)
	      {			/* state-dependent  */
		if (ret != 0)
		  {
		    /* non-zero: state-dependent encoding */
		    Result (C_SUCCESS, S_MBLEN, CASE_3, MS_PASSED);
		  }
		else
		  {
		    err_count++;
		    Result (C_FAILURE, S_MBLEN, CASE_3,
			    "should be state-dependent encoding, "
			    "but the return value shows it is"
			    " state-independent");
		  }
	      }

	    if (ret_exp == 0)
	      {			/* state-independent */
		if (ret == 0)
		  {
		    /* non-zero: state-dependent encoding */
		    Result (C_SUCCESS, S_MBLEN, CASE_3, MS_PASSED);
		  }
		else
		  {
		    err_count++;
		    Result (C_FAILURE, S_MBLEN, CASE_3,
			    "should be state-independent encoding, "
			    "but the return value shows it is"
			    " state-dependent");
		  }
	      }
	  }
      }
    }
  }

  return err_count;
}
