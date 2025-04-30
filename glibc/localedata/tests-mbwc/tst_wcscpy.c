/*-------------------------------------------------------------------------------------*/
/* WCSCPY: wchar_t *wcscpy( wchar_t *ws1, const wchar_t *ws2 )			       */
/*-------------------------------------------------------------------------------------*/
#define TST_FUNCTION wcscpy

#include "tsp_common.c"
#include "dat_wcscpy.c"

int
tst_wcscpy (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  wchar_t ws1[WCSSIZE], *ws2, *ws_ex;
  int err, i;

  TST_DO_TEST (wcscpy)
  {
    TST_HEAD_LOCALE (wcscpy, S_WCSCPY);
    TST_DO_REC (wcscpy)
    {
      TST_GET_ERRET (wcscpy);
      ws2 = TST_INPUT (wcscpy).ws;	/* external value: size WCSSIZE */
      ret = wcscpy (ws1, ws2);

      TST_IF_RETURN (S_WCSCPY)
      {
	if (ret == ws1)
	  {
	    Result (C_SUCCESS, S_WCSCPY, CASE_3, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_WCSCPY, CASE_3,
		    "the return address may not be correct");
	  }
      }

      if (ret == ws1)
	{
	  ws_ex = TST_EXPECT (wcscpy).ws;

	  for (err = 0, i = 0;
	       i < WCSSIZE && (ws1[i] != 0L || ws_ex[i] != 0L); i++)
	    {
	      if (debug_flg)
		{
		  fprintf (stderr,
			   "ws1[ %d ] = 0x%lx <-> wx_ex[ %d ] = 0x%lx\n", i,
			   (unsigned long int) ws1[i], i,
			   (unsigned long int) ws_ex[i]);
		}

	      if (ws1[i] != ws_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCSCPY, CASE_4,
			  "copied string is different from an"
			  " expected string");
		  break;
		}
	    }

	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCSCPY, CASE_4, MS_PASSED);
	    }

	  if (ws1[i] == 0L)
	    {
	      Result (C_SUCCESS, S_WCSCPY, CASE_5, MS_PASSED);
	    }
	  else
	    {
	      err_count++;
	      Result (C_FAILURE, S_WCSCPY, CASE_5,
		      "copied string is not null-terminated");
	    }
	}
    }
  }

  return err_count;
}
