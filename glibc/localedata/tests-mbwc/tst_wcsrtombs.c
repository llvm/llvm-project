/*
  WCSRTOMBS: size_t wcsrtombs (char *s, const wchar_t **ws, size_t n,
			       mbstate_t *ps)
*/

#define TST_FUNCTION wcsrtombs

#include "tsp_common.c"
#include "dat_wcsrtombs.c"

#define MARK_VAL 0x01

int
tst_wcsrtombs (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  char s_flg, n;
  const wchar_t *ws, *wp;
  char s[MBSSIZE], *s_in;
  char t_flg, t_ini;
  static mbstate_t t = { 0 };
  mbstate_t *pt;
  int err, i;
  char *s_ex;

  TST_DO_TEST (wcsrtombs)
  {
    TST_HEAD_LOCALE (wcsrtombs, S_WCSRTOMBS);
    TST_DO_REC (wcsrtombs)
    {
      TST_GET_ERRET (wcsrtombs);
      memset (s, MARK_VAL, MBSSIZE);

      s_flg = TST_INPUT (wcsrtombs).s_flg;
      s_in = (s_flg == 1) ? s : (char *) NULL;
      wp = ws = TST_INPUT (wcsrtombs).ws;
      n = TST_INPUT (wcsrtombs).n;
      t_flg = TST_INPUT (wcsrtombs).t_flg;
      t_ini = TST_INPUT (wcsrtombs).t_init;
      pt = (t_flg == 0) ? NULL : &t;

      if (t_ini != 0)
	{
	  memset (&t, 0, sizeof (t));
	}

      TST_CLEAR_ERRNO;
      ret = wcsrtombs (s_in, &wp, n, pt);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stderr, "wcsrtombs: ret	= %zu\n", ret);
	}

      TST_IF_RETURN (S_WCSRTOMBS)
      {
      };

      if (s_in != NULL && ret != (size_t) - 1)
	{
	  /* No definition for s, when error occurs.  */
	  s_ex = TST_EXPECT (wcsrtombs).s;

	  for (err = 0, i = 0; i <= ret && i < MBSSIZE; i++)
	    {
	      if (debug_flg)
		{
		  fprintf (stderr,
			   "	: s[%d] = 0x%hx <-> 0x%hx = s_ex[%d]\n", i,
			   s[i], s_ex[i], i);
		}

	      if (i == ret && ret == n)	/* no null termination */
		{
		  if (s[i] == MARK_VAL)
		    {
		      Result (C_SUCCESS, S_WCSRTOMBS, CASE_4, MS_PASSED);
		    }
		  else
		    {
		      err_count++;
		      Result (C_FAILURE, S_WCSRTOMBS, CASE_4,
			      "should not be null terminated "
			      "(it may be a null char), but it is");
		    }

		  break;
		}

	      if (i == ret && ret < n)	/* null termination */
		{
		  if (s[i] == 0)
		    {
		      Result (C_SUCCESS, S_WCSRTOMBS, CASE_5, MS_PASSED);
		    }
		  else
		    {
		      err_count++;
		      Result (C_FAILURE, S_WCSRTOMBS, CASE_5,
			      "should be null terminated, but it is not");
		    }

		  break;
		}

	      if (s[i] != s_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCSRTOMBS, CASE_6,
			  "converted string is different from an"
			  " expected string");
		  break;
		}
	    }

	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCSRTOMBS, CASE_6, MS_PASSED);
	    }
	}
    }
  }

  return err_count;
}
