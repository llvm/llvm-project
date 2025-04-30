/*
  MBTOWC: int mbtowc (wchar_t *wc, char *s, size_t n)
*/

#define TST_FUNCTION mbtowc

#include "tsp_common.c"
#include "dat_mbtowc.c"


int
tst_mbtowc (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  char w_flg, s_flg;
  const char *s_in;
  size_t n;
  wchar_t wc, wc_ex, *wp;

  TST_DO_TEST (mbtowc)
  {
    TST_HEAD_LOCALE (mbtowc, S_MBTOWC);
    TST_DO_REC (mbtowc)
    {
      if (mbstowcs (NULL, "", 0) != 0)
	{
	  err_count++;
	  Result (C_FAILURE, S_MBSTOWCS, CASE_3,
		  "Initialization failed - skipping this test case.");
	  continue;
	}

      TST_DO_SEQ (MBTOWC_SEQNUM)
      {
	TST_GET_ERRET_SEQ (mbtowc);
	w_flg = TST_INPUT_SEQ (mbtowc).w_flg;
	s_flg = TST_INPUT_SEQ (mbtowc).s_flg;
	n = TST_INPUT_SEQ (mbtowc).n;

	if (n == USE_MBCURMAX)
	  {
	    n = MB_CUR_MAX;
	  }

	if (s_flg == 0)
	  s_in = NULL;
	else
	  s_in = TST_INPUT_SEQ (mbtowc).s;

	wp = (wchar_t *) ((w_flg == 0) ? NULL : &wc);

	/* XXX Clear the internal state.  We should probably have
	   a flag for this.  */
	mbtowc (NULL, NULL, 0);

	TST_CLEAR_ERRNO;
	ret = mbtowc (wp, s_in, n);
	TST_SAVE_ERRNO;

	if (debug_flg)
	  {
	    fprintf (stdout, "mbtowc() [ %s : %d ] ret = %d\n", locale,
		     rec + 1, ret);
	    fprintf (stdout, "			   errno      = %d\n",
		     errno_save);
	  }

	TST_IF_RETURN (S_MBTOWC)
	{
	  if (s_in == NULL)
	    {			/* state dependency */
	      if (ret_exp == +1)
		{		/* state-dependent  */
		  if (ret != 0)
		    {
		      /* Non-zero: state-dependent encoding.  */
		      Result (C_SUCCESS, S_MBTOWC, CASE_3, MS_PASSED);
		    }
		  else
		    {
		      err_count++;
		      Result (C_FAILURE, S_MBTOWC, CASE_3,
			      "should be state-dependent encoding, "
			      "but a return value shows it is "
			      "state-independent");
		    }
		}

	      if (ret_exp == 0)
		{		/* state-independent */
		  if (ret == 0)
		    {
		      /* Non-zero: state-dependent encoding.  */
		      Result (C_SUCCESS, S_MBTOWC, CASE_3, MS_PASSED);
		    }
		  else
		    {
		      err_count++;
		      Result (C_FAILURE, S_MBTOWC, CASE_3,
			      "should be state-independent encoding, "
			      "but a return value shows it is "
			      "state-dependent");
		    }
		}
	    }
	}

	if ((wp == NULL || s_in == NULL || s_in[0] == 0) || ret <= 0)
	  {
	    continue;
	  }

	wc_ex = TST_EXPECT_SEQ (mbtowc).wc;

	if (wc_ex == wc)
	  {
	    Result (C_SUCCESS, S_MBTOWC, CASE_4, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_MBTOWC, CASE_4,
		    "converted wc is different from an expected wc");
	  }
      }
    }
  }

  return err_count;
}
