/*
  WCSLEN: size_t wcslen (const wchar_t *ws);
*/

#define TST_FUNCTION wcslen

#include "tsp_common.c"
#include "dat_wcslen.c"

int
tst_wcslen (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  wchar_t *ws;

  TST_DO_TEST (wcslen)
  {
    TST_HEAD_LOCALE (wcslen, S_WCSLEN);
    TST_DO_REC (wcslen)
    {
      TST_GET_ERRET (wcslen);
      ws = TST_INPUT (wcslen).ws;
      ret = wcslen (ws);
      TST_IF_RETURN (S_WCSLEN)
      {
      };
    }
  }

  return err_count;
}
