/*
  WCSCMP: int wcscmp (const wchar_t *ws1, const wchar_t *ws2);
*/

#define TST_FUNCTION wcscmp

#include "tsp_common.c"
#include "dat_wcscmp.c"


int
tst_wcscmp (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t *ws1, *ws2;

  TST_DO_TEST (wcscmp)
  {
    TST_HEAD_LOCALE (wcscmp, S_WCSCMP);
    TST_DO_REC (wcscmp)
    {
      TST_GET_ERRET (wcscmp);
      ws1 = TST_INPUT (wcscmp).ws1;
      ws2 = TST_INPUT (wcscmp).ws2;
      ret = wcscmp (ws1, ws2);
      ret = (ret > 0 ? 1 : ret < 0 ? -1 : 0);

      if (debug_flg)
	{
	  fprintf (stderr, "tst_wcscmp: ret = %d\n", ret);
	}

      TST_IF_RETURN (S_WCSCMP)
      {
      };
    }
  }

  return err_count;
}
