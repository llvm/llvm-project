/*
  WCSWIDTH: int wcswidth (const wchar_t *ws, size_t n);
*/

#define TST_FUNCTION wcswidth

#include "tsp_common.c"
#include "dat_wcswidth.c"

int
tst_wcswidth (FILE *fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t *ws;
  int n;

  TST_DO_TEST (wcswidth)
  {
    TST_HEAD_LOCALE (wcswidth, S_WCSWIDTH);
    TST_DO_REC (wcswidth)
    {
      TST_GET_ERRET (wcswidth);
      ws = TST_INPUT (wcswidth).ws;
      n = TST_INPUT (wcswidth).n;
      ret = wcswidth (ws, n);

      if (debug_flg)
	{
	  fprintf (stderr, "wcswidth: [ %d ] : ret = %d\n", rec + 1, ret);
	}

      TST_IF_RETURN (S_WCSWIDTH)
      {
      };
    }
  }

  return err_count;
}
