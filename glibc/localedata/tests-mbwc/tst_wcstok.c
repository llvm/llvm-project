/*
  WCSTOK: wchar_t *wcstok (wchar_t *ws, const wchar_t *dlm, wchar_t **pt);
*/


#define TST_FUNCTION wcstok

#include "tsp_common.c"
#include "dat_wcstok.c"

int
tst_wcstok (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  char w_flg;
  wchar_t *ws;
  wchar_t *dt, *pt;
  wchar_t *ws_ex;
  int err, i;

  TST_DO_TEST (wcstok)
  {
    TST_HEAD_LOCALE (wcstok, S_WCSTOK);
    TST_DO_REC (wcstok)
    {
      TST_DO_SEQ (WCSTOK_SEQNUM)
      {
	TST_GET_ERRET_SEQ (wcstok);
	w_flg = TST_INPUT_SEQ (wcstok).w_flg;
	ws = (w_flg) ? TST_INPUT_SEQ (wcstok).ws : NULL;
	dt = TST_INPUT_SEQ (wcstok).dt;

	ret = wcstok (ws, dt, &pt);

	if (debug_flg)
	  {
	    fprintf (stdout, "wcstok() [ %s : %d : %d ] *ret  = 0x%lx\n",
		     locale, rec + 1, seq_num + 1, (unsigned long int) *ret);
	    if (pt && *pt)
	      {
		fprintf (stdout, "			 *pt   = 0x%lx\n",
			 (unsigned long int) *pt);
	      }
	  }

	TST_IF_RETURN (S_WCSTOK)
	{
	};

	if (ret != NULL)
	  {
	    ws_ex = TST_EXPECT_SEQ (wcstok).ws;

	    /* XXX: REVISIT : insufficient conditions */
	    for (err = 0, i = 0; i < WCSSIZE; i++)
	      {
		if (ret[i] == L'\0' && ws_ex[i] == L'\0')
		  {
		    break;
		  }

		if (debug_flg)
		  {
		    fprintf (stderr,
			     "			      ret[%d] = 0x%lx <-> "
			     "0x%lx = ws_ex[%d]\n",
			     i, (unsigned long int) ret[i],
			     (unsigned long int) ws_ex[i], i);
		  }

		if (ret[i] != ws_ex[i])
		  {
		    err++;
		    err_count++;
		    Result (C_FAILURE, S_WCSTOK, CASE_3,
			    "the token is different from an expected string");
		    break;
		  }

		if (ret[i] == L'\0' || ws_ex[i] == L'\0')
		  {
		    break;
		  }
	      }

	    if (!err)
	      {
		Result (C_SUCCESS, S_WCSTOK, CASE_3, MS_PASSED);
	      }
	  }
      }
    }
  }

  return err_count;
}
