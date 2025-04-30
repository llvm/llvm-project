/*
  WCSPBRK: wchar_t *wcspbrk (const wchar_t *ws1, const wchar_t *ws2);
*/

#define TST_FUNCTION wcspbrk

#include "tsp_common.c"
#include "dat_wcspbrk.c"

int
tst_wcspbrk (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  wchar_t *ws1, *ws2;
  int err;
  wchar_t wc_ex;

  TST_DO_TEST (wcspbrk)
  {
    TST_HEAD_LOCALE (wcspbrk, S_WCSPBRK);
    TST_DO_REC (wcspbrk)
    {
      TST_GET_ERRET (wcspbrk);
      ws1 = TST_INPUT (wcspbrk).ws1;
      ws2 = TST_INPUT (wcspbrk).ws2;

      ret = wcspbrk (ws1, ws2);

      if (debug_flg)
	{
	  fprintf (stdout, "wcspbrk() [ %s : %d ] ret = %s\n", locale,
		   rec + 1, (ret == NULL) ? "null" : "not null");
	  if (ret)
	    fprintf (stderr,
		     "			      ret[0] = 0x%lx : 0x%lx = ws2[0]\n",
		     (unsigned long int) ret[0], (unsigned long int) ws2[0]);
	}

      TST_IF_RETURN (S_WCSPBRK)
      {
	if (ws2[0] == 0)
	  {
	    if (ret == ws1)
	      {
		Result (C_SUCCESS, S_WCSPBRK, CASE_3, MS_PASSED);
	      }
	    else
	      {
		err_count++;
		Result (C_FAILURE, S_WCSPBRK, CASE_3,
			"return address is not same address as ws1");
	      }

	    continue;
	  }

	wc_ex = TST_EXPECT (wcspbrk).wc;

	if (debug_flg)
	  fprintf (stdout,
		   "			    *ret = 0x%lx <-> 0x%lx = wc_ex\n",
		   (unsigned long int) *ret, (unsigned long int) wc_ex);

	if (*ret != wc_ex)
	  {
	    err++;
	    err_count++;
	    Result (C_FAILURE, S_WCSPBRK, CASE_4, "the pointed wc is "
		    "different from an expected wc");
	  }
	else
	  {
	    Result (C_SUCCESS, S_WCSPBRK, CASE_4, MS_PASSED);
	  }
      }
    }
  }

  return err_count;
}
