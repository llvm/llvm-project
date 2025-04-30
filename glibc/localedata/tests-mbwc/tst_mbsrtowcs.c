/*
  MBSRTOWCS: size_t mbsrtowcs (wchar_t *ws, const char **s, size_t n,
			       mbstate_t *ps)
*/

#define TST_FUNCTION mbsrtowcs

#include "tsp_common.c"
#include "dat_mbsrtowcs.c"

int
tst_mbsrtowcs (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  char w_flg;
  const char *s, *p;
  size_t n;
  char t_flg, t_ini;
  static mbstate_t t = { 0 };
  mbstate_t *pt;
  wchar_t ws[WCSSIZE], *ws_ex, *wp;
  int err, i;

  TST_DO_TEST (mbsrtowcs)
  {
    TST_HEAD_LOCALE (mbsrtowcs, S_MBSRTOWCS);
    TST_DO_REC (mbsrtowcs)
    {
      s = "";
      if (mbsrtowcs (NULL, &s, 0, &t) != 0)
	{
	  err_count++;
	  Result (C_FAILURE, S_MBSRTOWCS, CASE_3,
		  "Initialization failed - skipping this test case.");
	  continue;
	}

      TST_DO_SEQ (MBSRTOWCS_SEQNUM)
      {
	TST_GET_ERRET_SEQ (mbsrtowcs);
	w_flg = TST_INPUT_SEQ (mbsrtowcs).w_flg;
	p = s = TST_INPUT_SEQ (mbsrtowcs).s;
	n = TST_INPUT_SEQ (mbsrtowcs).n;
	t_flg = TST_INPUT_SEQ (mbsrtowcs).t_flg;
	t_ini = TST_INPUT_SEQ (mbsrtowcs).t_init;
	wp = (w_flg == 0) ? NULL : ws;

	if (n == USE_MBCURMAX)
	  {
	    n = MB_CUR_MAX;
	  }

	pt = (t_flg == 0) ? NULL : &t;

	if (t_ini != 0)
	  {
	    memset (&t, 0, sizeof (t));
	  }

	TST_CLEAR_ERRNO;
	ret = mbsrtowcs (wp, &p, n, pt);
	TST_SAVE_ERRNO;

	if (debug_flg)
	  {
	    fprintf (stderr, "mbsrtowcs: [ %d ] : ret = %zd\n", rec + 1, ret);
	  }

	TST_IF_RETURN (S_MBSRTOWCS)
	{
	};

	if (wp == NULL || ret == (size_t) - 1 || ret == (size_t) - 2)
	  {
	    continue;
	  }

	ws_ex = TST_EXPECT_SEQ (mbsrtowcs).ws;
	for (err = 0, i = 0; i < ret; i++)
	  {
	    if (debug_flg)
	      {
		fprintf (stderr,
			 "mbsrtowcs: ws[%d] => 0x%lx : 0x%lx <= ws_ex[%d]\n",
			 i, (unsigned long int) ws[i],
			 (unsigned long int) ws_ex[i], i);
	      }

	    if (ws[i] != ws_ex[i])
	      {
		err++;
		err_count++;
		Result (C_FAILURE, S_MBSRTOWCS, CASE_4,
			"the converted wc string has "
			"different value from an expected string");
		break;
	      }
	  }

	if (!err)
	  {
	    Result (C_SUCCESS, S_MBSRTOWCS, CASE_4, MS_PASSED);
	  }
      }
    }
  }

  return err_count;
}
