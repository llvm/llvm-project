/*
  WCRTOMB: wchar_t wcrtomb (char *s, wchar_t wc, mbstate_t *ps)
*/

#define TST_FUNCTION wcrtomb

#include "tsp_common.c"
#include "dat_wcrtomb.c"


int
tst_wcrtomb (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wchar_t);
  wchar_t wc;
  char s[MBSSIZE], *s_in, *s_ex;
  char t_flg, t_ini;
  static mbstate_t t = { 0 };
  mbstate_t *pt;
  int err, i;

  TST_DO_TEST (wcrtomb)
  {
    TST_HEAD_LOCALE (wcrtomb, S_WCRTOMB);
    TST_DO_REC (wcrtomb)
    {
      TST_GET_ERRET (wcrtomb);
      s_in = ((TST_INPUT (wcrtomb).s_flg) == 0) ? (char *) NULL : s;
      wc = TST_INPUT (wcrtomb).wc;
      t_flg = TST_INPUT (wcrtomb).t_flg;
      t_ini = TST_INPUT (wcrtomb).t_init;
      pt = (t_flg == 0) ? NULL : &t;

      if (t_ini != 0)
	{
	  memset (&t, 0, sizeof (t));
	}

      TST_CLEAR_ERRNO;
      ret = wcrtomb (s_in, wc, pt);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stdout, "wcrtomb() [ %s : %d ] ret = %lu\n", locale,
		   rec + 1, (unsigned long int) ret);
	  fprintf (stdout, "			errno = %d\n", errno_save);
	}

      TST_IF_RETURN (S_WCRTOMB)
      {
      };

      s_ex = TST_EXPECT (wcrtomb).s;

      if (s_in)
	{
	  for (i = 0, err = 0; *(s_ex + i) != 0 && i < MBSSIZE; i++)
	    {
	      if (s_in[i] != s_ex[i])
		{
		  err++;
		  err_count++;
		  Result (C_FAILURE, S_WCRTOMB, CASE_4,
			  "copied string is different from an "
			  "expected string");
		  break;
		}
	    }
	  if (!err)
	    {
	      Result (C_SUCCESS, S_WCRTOMB, CASE_4, MS_PASSED);
	    }
	}
    }
  }

  return err_count;
}
