/*
  WCSNCAT: wchar_t *wcsncat (wchar_t *ws1, const wchar_t *ws2, size_t n);
*/

#define TST_FUNCTION wcsncat

#include "tsp_common.c"
#include "dat_wcsncat.c"

int
tst_wcsncat (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  wchar_t *ws1, *ws2, *ws_ex;
  int n, i, err;

  TST_DO_TEST (wcsncat)
  {
    TST_HEAD_LOCALE (wcsncat, S_WCSNCAT);
    TST_DO_REC (wcsncat)
    {
      TST_GET_ERRET (wcsncat);
      ws1 = TST_INPUT (wcsncat).ws1;	/* external value: size WCSSIZE */
      ws2 = TST_INPUT (wcsncat).ws2;
      n = TST_INPUT (wcsncat).n;
      ret = wcsncat (ws1, ws2, n);

      TST_IF_RETURN (S_WCSNCAT)
      {
	if (ret == ws1)
	  {
	    Result (C_SUCCESS, S_WCSNCAT, CASE_3, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_WCSNCAT, CASE_3,
		    "the return address may not be correct");
	  }
      }

      if (ret == ws1)
	{
	  ws_ex = TST_EXPECT (wcsncat).ws;

	  for (err = 0, i = 0;
	       (ws1[i] != 0L || ws_ex[i] != 0L) && i < WCSSIZE; i++)
	    {
	      if (debug_flg)
		{
		  fprintf (stderr, "ws1[%d] = 0x%lx\n", i,
			   (unsigned long int) ws1[i]);
		}

	      if (ws1[i] != ws_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCSNCAT, CASE_4,
			  "the concatinated string has "
			  "different value from an expected string");
		  break;
		}
	    }

	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCSNCAT, CASE_4, MS_PASSED);
	    }
	}
    }
  }

  return err_count;
}
