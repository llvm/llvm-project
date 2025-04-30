/*
  WCSSTR: wchar_t *wcsstr (const wchar_t *ws1, const wchar_t *ws2);
*/

#define TST_FUNCTION wcsstr

#include "tsp_common.c"
#include "dat_wcsstr.c"

int
tst_wcsstr (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t *);
  wchar_t *ws1, *ws2;
  int err, i;

  TST_DO_TEST (wcsstr)
  {
    TST_HEAD_LOCALE (wcsstr, S_WCSSTR);
    TST_DO_REC (wcsstr)
    {
      TST_GET_ERRET (wcsstr);
      ws1 = TST_INPUT (wcsstr).ws1;
      ws2 = TST_INPUT (wcsstr).ws2;	/* external value: size WCSSIZE */
      ret = wcsstr (ws1, ws2);

      if (debug_flg)
	{
	  fprintf (stderr, "wcsstr: %d : ret = %s\n", rec + 1,
		   (ret == NULL) ? "null" : "not null");
	  if (ret)
	    {
	      fprintf (stderr,
		       "	ret[ 0 ] = 0x%lx <-> 0x%lx = ws2[ 0 ]\n",
		       (unsigned long int) ret[0], (unsigned long int) ws2[0]);
	    }
	}

      TST_IF_RETURN (S_WCSSTR)
      {
	if (ws2[0] == 0)
	  {
	    if (ret == ws1)
	      {
		Result (C_SUCCESS, S_WCSSTR, CASE_3, MS_PASSED);
	      }
	    else
	      {
		err_count++;
		Result (C_FAILURE, S_WCSSTR, CASE_3,
			"return address is not same address as ws1");
	      }

	    continue;
	  }

	for (i = 0, err = 0; *(ws2 + i) != 0 && i < WCSSIZE; i++)
	  {
	    if (debug_flg)
	      {
		fprintf (stderr,
			 "	: ret[ %d ] = 0x%lx <-> 0x%lx = ws2[ %d ]\n",
			 i, (unsigned long int) ret[i],
			 (unsigned long int) ws2[i], i);
	      }

	    if (ret[i] != ws2[i])
	      {
		err++;
		err_count++;
		Result (C_FAILURE, S_WCSSTR, CASE_4, "pointed sub-string is "
			"different from an expected sub-string");
		break;
	      }
	  }

	if (!err)
	  {
	    Result (C_SUCCESS, S_WCSSTR, CASE_4, MS_PASSED);
	  }
      }
    }
  }

  return err_count;
}
