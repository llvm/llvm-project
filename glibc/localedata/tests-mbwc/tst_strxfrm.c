/*
  STRXFRM: size_t strxfrm (char *s1, const char *s2, size_t n)
*/

#define TST_FUNCTION strxfrm

#include "tsp_common.c"
#include "dat_strxfrm.c"


int
tst_strxfrm (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (size_t);
  const char *org1, *org2;
  char frm1[MBSSIZE], frm2[MBSSIZE];
  size_t n1, n2;
  int ret_coll, ret_cmp;

  TST_DO_TEST (strxfrm)
  {
    TST_HEAD_LOCALE (strxfrm, S_STRXFRM);
    TST_DO_REC (strxfrm)
    {
      TST_GET_ERRET (strxfrm);
      org1 = TST_INPUT (strxfrm).org1;
      org2 = TST_INPUT (strxfrm).org2;
      n1 = TST_INPUT (strxfrm).n1;
      n2 = TST_INPUT (strxfrm).n2;

      if (n1 < 0 || sizeof (frm1) < n1 || sizeof (frm2) < n2)
	{
	  warn_count++;
	  Result (C_IGNORED, S_STRXFRM, CASE_9,
		  "input data n1 or n2 is invalid");
	  continue;
	}

      /* An errno and a return value are checked
	 only for 2nd strxfrm() call.
	 A result of 1st call is used for comparing
	 those 2 values by using strcmp().
      */

      /*-- First call --*/

      TST_CLEAR_ERRNO;
      ret = strxfrm (frm1, org1, n1);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stdout, "strxfrm() [ %s : %d ] ( 1st call )\n", locale,
		   rec + 1);
	  fprintf (stdout, "	  : err = %d | %s\n", errno_save,
		   strerror (errno));
	  fprintf (stdout, "	  : ret = %zu\n", ret);
	  fprintf (stdout, "	  : org = %s\n", org1);
	}

      if (ret >= n1 || errno != 0)
	{
	  warn_count++;
	  Result (C_INVALID, S_STRXFRM, CASE_8,
		  "got an error in fist strxfrm() call");
	  continue;
	}

      /*-- Second call --*/

      TST_CLEAR_ERRNO;
      ret = strxfrm (((n2 == 0) ? NULL : frm2), org2, n2);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stderr, "	  ..............( 2nd call )\n");
	  fprintf (stdout, "	  : err = %d | %s\n", errno,
		   strerror (errno));
	  fprintf (stdout, "	  : ret = %zu\n", ret);
	  fprintf (stdout, "	  : org = %s\n", org2);
	}

      TST_IF_RETURN (S_STRXFRM)
      {
      };

      if (n2 == 0 || ret >= n2 || errno != 0)
	{
#if 0
	  warn_count++;
	  Result (C_IGNORED, S_STRXFRM, CASE_7, "did not get a result");
#endif
	  continue;
	}

      /*-- strcoll & strcmp --*/

      TST_CLEAR_ERRNO;
      /* Depends on strcoll() ... not good though ... */
      ret_coll = strcoll (org1, org2);

      if (errno != 0)
	{
	  /* bug * bug may get correct results ...	  */
	  warn_count++;
	  Result (C_INVALID, S_STRXFRM, CASE_6,
		  "got an error in strcoll() call");
	  continue;
	}

      ret_cmp = strcmp (frm1, frm2);

      if ((ret_coll == 0 && ret_cmp == 0)
	  || (ret_coll < 0 && ret_cmp < 0) || (ret_coll > 0 && ret_cmp > 0))
	{
	  Result (C_SUCCESS, S_STRXFRM, CASE_3,
		  MS_PASSED "(depends on strcoll & strcmp)");
	}
      else
	{
	  err_count++;
	  Result (C_FAILURE, S_STRXFRM, CASE_3,
		  "results from strcoll & strcmp() do not match");
	}

      if (debug_flg)
	{
	  fprintf (stdout, ".......... strcoll = %d <-> %d = strcmp\n",
		   ret_coll, ret_cmp);
	}
    }
  }

  return err_count;
}
