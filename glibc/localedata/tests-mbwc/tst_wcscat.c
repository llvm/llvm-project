/*
  WCSCAT: wchar_t *wcscat (wchar_t *ws1, const wchar_t *ws2);
*/

#define TST_FUNCTION wcscat

#include "tsp_common.c"
#include "dat_wcscat.c"

int
tst_wcscat (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  wchar_t *ws1, *ws2, *ws_ex;
  int i, err;

  TST_DO_TEST (wcscat)
  {
    TST_HEAD_LOCALE (wcscat, S_WCSCAT);
    TST_DO_REC (wcscat)
    {
      TST_GET_ERRET (wcscat);
      ws1 = TST_INPUT (wcscat).ws1;	/* external value: size WCSSIZE */
      ws2 = TST_INPUT (wcscat).ws2;

      TST_CLEAR_ERRNO;
      ret = wcscat (ws1, ws2);
      TST_SAVE_ERRNO;

      TST_IF_RETURN (S_WCSCAT)
      {
	if (ret == ws1)
	  {
	    Result (C_SUCCESS, S_WCSCAT, CASE_3, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_WCSCAT, CASE_3,
		    "the return address may not be correct");
	  }
      }

      /* function specific test cases here */

      if (ret == ws1)
	{
	  ws_ex = TST_EXPECT (wcscat).ws;
	  for (err = 0, i = 0;
	       (ws1[i] != 0L || ws_ex[i] != 0L) && i < WCSSIZE; i++)
	    {
	      if (debug_flg)
		{
		  fprintf (stdout, "tst_wcscat() : ws1[%d] = 0x%lx\n", i,
			   (unsigned long int) ws1[i]);
		}

	      if (ws1[i] != ws_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCSCAT, CASE_4,
			  "concatinated string is different from an "
			  "expected string");
		  break;
		}
	    }

	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCSCAT, CASE_4, MS_PASSED);
	    }
	}
    }
  }

  return err_count;
}
