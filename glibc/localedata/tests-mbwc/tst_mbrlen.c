/*
  MBRLEN: size_t mbrlen (char *s, size_t n, mbstate_t *ps)
*/

#define TST_FUNCTION mbrlen

#include "tsp_common.c"
#include "dat_mbrlen.c"


int
tst_mbrlen (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  char s_flg;
  const char *s_in;
  size_t n;
  char t_flg;
  char t_ini;
  static mbstate_t s = { 0 };
  mbstate_t *ps;

  TST_DO_TEST (mbrlen)
  {
    TST_HEAD_LOCALE (mbrlen, S_MBRLEN);
    TST_DO_REC (mbrlen)
    {
      if (mbrlen (NULL, 0, &s) != 0)
	{
	  err_count++;
	  Result (C_FAILURE, S_MBRLEN, CASE_3,
		  "Initialization (external mbstate object) failed "
		  "- skipped this test case.");
	  continue;
	}

      TST_DO_SEQ (MBRLEN_SEQNUM)
      {
	TST_GET_ERRET_SEQ (mbrlen);
	s_flg = TST_INPUT_SEQ (mbrlen).s_flg;
	s_in = TST_INPUT_SEQ (mbrlen).s;
	n = TST_INPUT_SEQ (mbrlen).n;
	t_flg = TST_INPUT_SEQ (mbrlen).t_flg;
	t_ini = TST_INPUT_SEQ (mbrlen).t_init;
	if (s_flg == 0)
	  {
	    s_in = NULL;
	  }

	if (n == USE_MBCURMAX)	/* rewrite tst_mblen() like this */
	  {
	    n = MB_CUR_MAX;
	  }

	ps = (t_flg == 0) ? NULL : &s;

	if (t_ini != 0)
	  {
	    memset (&s, 0, sizeof (s));
	    mbrlen (NULL, 0, NULL);
	  }

	TST_CLEAR_ERRNO;
	ret = mbrlen (s_in, n, ps);
	TST_SAVE_ERRNO;

	if (debug_flg)
	  {
	    fprintf (stdout, "mbrlen() [ %s : %d : %d ] ret = %zd\n",
		     locale, rec + 1, seq_num + 1, ret);
	    fprintf (stdout, "			   errno = %d\n", errno_save);
	  }

	TST_IF_RETURN (S_MBRLEN)
	{
	};
      }
    }
  }

  return err_count;
}
