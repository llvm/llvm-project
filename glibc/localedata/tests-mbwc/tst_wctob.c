/*-------------------------------------------------------------------------------------*/
/* WCTOB: int wctob( wint_t wc )						       */
/*-------------------------------------------------------------------------------------*/

#define TST_FUNCTION wctob

#include "tsp_common.c"
#include "dat_wctob.c"

int
tst_wctob (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t wc;

  TST_DO_TEST (wctob)
  {
    TST_HEAD_LOCALE (wctob, S_WCTOB);
    TST_DO_REC (wctob)
    {
      TST_GET_ERRET (wctob);
      wc = TST_INPUT (wctob).wc;
      ret = wctob (wc);

      if (debug_flg)
	{
	  fprintf (stderr, "tst_wctob : [ %d ] ret = %d\n", rec + 1, ret);
	}

      TST_IF_RETURN (S_WCTOB)
      {
      };
    }
  }

  return err_count;
}
