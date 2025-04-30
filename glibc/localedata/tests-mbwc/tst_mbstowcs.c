/*
  MBSTOWCS: size_t mbstowcs (wchar_t *ws, char *s, size_t n)
*/

#define TST_FUNCTION mbstowcs

#include "tsp_common.c"
#include "dat_mbstowcs.c"

int
tst_mbstowcs (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  char w_flg, s_flg;
  const char *s;
  size_t n;
  wchar_t ws[WCSSIZE], *ws_ex, *wp;
  int err, i;

  TST_DO_TEST (mbstowcs)
  {
    TST_HEAD_LOCALE (mbstowcs, S_MBSTOWCS);
    TST_DO_REC (mbstowcs)
    {
      if (mbstowcs (NULL, "", 0) != 0)
	{
	  err_count++;
	  Result (C_FAILURE, S_MBSTOWCS, CASE_3,
		  "Initialization failed - skipping this test case.");
	  continue;
	}

      TST_DO_SEQ (MBSTOWCS_SEQNUM)
      {
	TST_GET_ERRET_SEQ (mbstowcs);
	w_flg = TST_INPUT_SEQ (mbstowcs).w_flg;
	s_flg = TST_INPUT_SEQ (mbstowcs).s_flg;
	n = TST_INPUT_SEQ (mbstowcs).n;

	if (s_flg == 0)
	  s = NULL;
	else
	  s = TST_INPUT_SEQ (mbstowcs).s;


	wp = (wchar_t *) ((w_flg == 0) ? NULL : ws);

	TST_CLEAR_ERRNO;
	ret = mbstowcs (wp, s, n);
	TST_SAVE_ERRNO;

	if (debug_flg)
	  {
	    fprintf (stderr, "mbstowcs: ret = %zd\n", ret);
	  }

	TST_IF_RETURN (S_MBSTOWCS)
	{
	};

	if (s == NULL || wp == NULL || ret == (size_t) - 1)
	  {
	    continue;
	  }

	ws_ex = TST_EXPECT_SEQ (mbstowcs).ws;

	for (err = 0, i = 0; i < ret; i++)
	  {
	    if (debug_flg)
	      {
		fprintf (stderr,
			 "mbstowcs: ws[%d] => 0x%lx : 0x%lx <= ws_ex[%d]\n",
			 i, (unsigned long int) ws[i],
			 (unsigned long int) ws_ex[i], i);
	      }

	    if (ws[i] != ws_ex[i])
	      {
		err++;
		err_count++;
		Result (C_FAILURE, S_MBSTOWCS, CASE_4,
			"the converted wc string has "
			"different value from an expected string");
		break;
	      }
	  }

	if (!err)
	  {
	    Result (C_SUCCESS, S_MBSTOWCS, CASE_4, MS_PASSED);
	  }
      }
    }
  }

  return err_count;
}
