/*
  WCSSPN: size_t wcsspn (const wchar_t *ws1, const wchar_t *ws2);
*/

#define TST_FUNCTION wcsspn

#include "tsp_common.c"
#include "dat_wcsspn.c"

int
tst_wcsspn (FILE *fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  wchar_t *ws1, *ws2;

  TST_DO_TEST (wcsspn)
  {
    TST_HEAD_LOCALE (wcsspn, S_WCSSPN);
    TST_DO_REC (wcsspn)
    {
      TST_GET_ERRET (wcsspn);
      ws1 = TST_INPUT (wcsspn).ws1;
      ws2 = TST_INPUT (wcsspn).ws2;	/* external value: size WCSSIZE */
      ret = wcsspn (ws1, ws2);

      if (debug_flg)
	{
	  fprintf (stderr, "wcsspn: ret = %zu\n", ret);
	}

      TST_IF_RETURN (S_WCSSPN)
      {
      };
    }
  }

  return err_count;
}
