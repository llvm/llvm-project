/*-------------------------------------------------------------------------------------*/
/* WCSNCMP: int wcsncmp( const wchar_t *ws1, const wchar_t *ws2, size_t n )	       */
/*-------------------------------------------------------------------------------------*/

#define TST_FUNCTION wcsncmp

#include "tsp_common.c"
#include "dat_wcsncmp.c"

int
tst_wcsncmp (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t *ws1, *ws2;
  size_t n;

  TST_DO_TEST (wcsncmp)
  {
    TST_HEAD_LOCALE (wcsncmp, S_WCSNCMP);
    TST_DO_REC (wcsncmp)
    {
      TST_GET_ERRET (wcsncmp);
      ws1 = TST_INPUT (wcsncmp).ws1;	/* external value: size WCSSIZE */
      ws2 = TST_INPUT (wcsncmp).ws2;
      n = TST_INPUT (wcsncmp).n;
      ret = wcsncmp (ws1, ws2, n);
      ret = (ret > 0 ? 1 : ret < 0 ? -1 : 0);

      if (debug_flg)
	{
	  fprintf (stderr, "tst_wcsncmp: ret = %d, 0x%x\n", ret, ret);
	}

      TST_IF_RETURN (S_WCSNCMP)
      {
      };
    }
  }

  return err_count;
}
