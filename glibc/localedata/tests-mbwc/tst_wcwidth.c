/*
  WCWIDTH: int wcwidth (wchar_t wc);
*/

#define TST_FUNCTION wcwidth

#include "tsp_common.c"
#include "dat_wcwidth.c"

int
tst_wcwidth (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t wc;

  TST_DO_TEST (wcwidth)
  {
    TST_HEAD_LOCALE (wcwidth, S_WCWIDTH);
    TST_DO_REC (wcwidth)
    {
      TST_GET_ERRET (wcwidth);
      wc = TST_INPUT (wcwidth).wc;
      ret = wcwidth (wc);

      if (debug_flg)
	{
	  fprintf (stdout, "wcwidth() [ %s : %d ] ret  = %d\n", locale,
		   rec + 1, ret);
	}

      TST_IF_RETURN (S_WCWIDTH)
      {
      }
    }
  }

  return err_count;
}
