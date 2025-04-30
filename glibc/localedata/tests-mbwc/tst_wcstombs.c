/*
  WCSTOMBS: size_t wcstombs (char *s, const wchar_t *ws, size_t n)
*/

#define TST_FUNCTION wcstombs

#include "tsp_common.c"
#include "dat_wcstombs.c"

#define MARK_VAL 0x01

int
tst_wcstombs (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  char s_flg, n;
  wchar_t *ws;
  char s[MBSSIZE], *s_in;
  int err, i;
  char *s_ex;

  TST_DO_TEST (wcstombs)
  {
    TST_HEAD_LOCALE (wcstombs, S_WCSTOMBS);
    TST_DO_REC (wcstombs)
    {
      TST_GET_ERRET (wcstombs);
      memset (s, MARK_VAL, MBSSIZE);

      s_flg = TST_INPUT (wcstombs).s_flg;
      s_in = (s_flg == 1) ? s : (char *) NULL;
      ws = TST_INPUT (wcstombs).ws;
      n = TST_INPUT (wcstombs).n;

      TST_CLEAR_ERRNO;
      ret = wcstombs (s_in, ws, n);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stdout, "wcstombs: ret  = %zu\n", ret);
	}

      TST_IF_RETURN (S_WCSTOMBS)
      {
      };

      if (s_in != NULL && ret != (size_t) - 1)
	{
	  /* No definition for s, when error occurs.  */
	  s_ex = TST_EXPECT (wcstombs).s;

	  for (err = 0, i = 0; i <= ret && i < MBSSIZE; i++)
	    {
	      if (debug_flg)
		{
		  fprintf (stdout,
			   "	: s[%d] = 0x%hx <-> 0x%hx = s_ex[%d]\n", i,
			   s[i], s_ex[i], i);
		}

	      if (i == ret && ret == n)	/* no null termination */
		{
		  if (s[i] == MARK_VAL)
		    {
		      Result (C_SUCCESS, S_WCSTOMBS, CASE_4, MS_PASSED);
		    }
		  else
		    {
		      err_count++;
		      Result (C_FAILURE, S_WCSTOMBS, CASE_4,
			      "should not be null terminated "
			      "(it may be a null char), but it is");
		    }

		  break;
		}

	      if (i == ret && ret < n)	/* null termination */
		{
		  if (s[i] == 0)
		    {
		      Result (C_SUCCESS, S_WCSTOMBS, CASE_5, MS_PASSED);
		    }
		  else
		    {
		      err_count++;
		      Result (C_FAILURE, S_WCSTOMBS, CASE_5,
			      "should be null terminated, but it is not");
		    }

		  break;
		}

	      if (s[i] != s_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCSTOMBS, CASE_6,
			  "converted string is different from an "
			  "expected string");
		  break;
		}
	    }

	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCSTOMBS, CASE_6, MS_PASSED);
	    }
	}
    }
  }

  return err_count;
}
