/*
  MBRTOWC: size_t mbrtowc (wchar_t *pwc, const char *s, size_t n,
			   mbstate_t *ps)
*/

#define TST_FUNCTION mbrtowc

#include "tsp_common.c"
#include "dat_mbrtowc.c"


int
tst_mbrtowc (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  char w_flg, s_flg;
  char *s;
  size_t n;
  char t_flg;
  static mbstate_t t = { 0 };
  mbstate_t *pt;
  wchar_t wc, *pwc, wc_ex;

  TST_DO_TEST (mbrtowc)
  {
    TST_HEAD_LOCALE (mbrtowc, S_MBRTOWC);
    TST_DO_REC (mbrtowc)
    {
      if (mbrtowc (NULL, "", 0, &t) != -2)
	{
	  err_count++;
	  Result (C_FAILURE, S_MBRTOWC, CASE_3,
		  "Initialization failed - skipping this test case.");
	  continue;
	}

      TST_DO_SEQ (MBRTOWC_SEQNUM)
      {
	TST_GET_ERRET_SEQ (mbrtowc);
	w_flg = TST_INPUT_SEQ (mbrtowc).w_flg;
	s_flg = TST_INPUT_SEQ (mbrtowc).s_flg;
	s = TST_INPUT_SEQ (mbrtowc).s;
	n = TST_INPUT_SEQ (mbrtowc).n;
	t_flg = TST_INPUT_SEQ (mbrtowc).t_flg;
	pwc = (w_flg == 0) ? NULL : &wc;

	if (s_flg == 0)
	  {
	    s = NULL;
	  }

	if (n == USE_MBCURMAX)
	  {
	    n = MB_CUR_MAX;
	  }

	pt = (t_flg == 0) ? NULL : &t;

	TST_CLEAR_ERRNO;
	ret = mbrtowc (pwc, s, n, pt);
	TST_SAVE_ERRNO;

	if (debug_flg)
	  {
	    fprintf (stdout, "mbrtowc() [ %s : %d : %d ] ret = %zd\n",
		     locale, rec + 1, seq_num + 1, ret);
	    fprintf (stdout, "			    errno = %hd\n",
		     errno_save);
	  }

	TST_IF_RETURN (S_MBRTOWC)
	{
	};

	if (pwc == NULL || s == NULL || ret == (size_t) - 1
	    || ret == (size_t) - 2)
	  {
	    continue;
	  }

	wc_ex = TST_EXPECT_SEQ (mbrtowc).wc;
	if (wc_ex == wc)
	  {
	    Result (C_SUCCESS, S_MBRTOWC, CASE_4, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_MBRTOWC, CASE_4,
		    "converted wc is different from an expected wc");
	  }
      }
    }
  }

  return err_count;
}
