/*
  WCSNCPY: wchar_t *wcsncpy (wchar_t *ws1, const wchar_t *ws2, size_t n);
*/

#define TST_FUNCTION wcsncpy

#include "tsp_common.c"
#include "dat_wcsncpy.c"

#define WCSNUM_NCPY 7

int
tst_wcsncpy (FILE *fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  wchar_t ws1[WCSSIZE] =
    { 0x9999, 0x9999, 0x9999, 0x9999, 0x9999, 0x9999, 0x0000 };
  wchar_t *ws2, *ws_ex;
  int err, i;
  size_t n;

  TST_DO_TEST (wcsncpy)
  {
    TST_HEAD_LOCALE (wcsncpy, S_WCSNCPY);
    TST_DO_REC (wcsncpy)
    {
      TST_GET_ERRET (wcsncpy);

      for (n = 0; n < WCSNUM_NCPY - 1; ++n)
	{
	  ws1[n] = 0x9999;
	}

      ws1[n] = 0;
      ws2 = TST_INPUT (wcsncpy).ws;	/* external value: size WCSSIZE */
      n = TST_INPUT (wcsncpy).n;
      ret = wcsncpy (ws1, ws2, n);

      TST_IF_RETURN (S_WCSNCPY)
      {
	if (ret == ws1)
	  {
	    Result (C_SUCCESS, S_WCSNCPY, CASE_3, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_WCSNCPY, CASE_3,
		    "the return address may not be correct");
	  }
      }

      if (ret == ws1)
	{
	  if (debug_flg)
	    {
	      fprintf (stderr, "\nwcsncpy: n = %zu\n\n", n);
	    }

	  ws_ex = TST_EXPECT (wcsncpy).ws;

	  for (err = 0, i = 0; i < WCSNUM_NCPY && i < WCSSIZE; i++)
	    {
	      if (debug_flg)
		fprintf (stderr,
			 "wcsncpy: ws1[ %d ] = 0x%lx <-> wx_ex[ %d ] = 0x%lx\n",
			 i, (unsigned long int) ws1[i], i,
			 (unsigned long int) ws_ex[i]);

	      if (ws1[i] != ws_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCSNCPY, CASE_4,
			  "copied string is different from an "
			  "expected string");
		  break;
		}
	    }

	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCSNCPY, CASE_4, MS_PASSED);
	    }

	  /* A null terminate character is not supposed to be copied
	     unless (num chars of ws2)<n. */
	}
    }
  }

  return err_count;
}
