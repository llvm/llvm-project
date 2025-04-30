/*
  WCTYPE: wctype_t wctype (const char *class);
*/


#define TST_FUNCTION wctype

#include "tsp_common.c"
#include "dat_wctype.c"

int
tst_wctype (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (wctype_t);
  char *class;

  TST_DO_TEST (wctype)
  {
    TST_HEAD_LOCALE (wctype, S_WCTYPE);
    TST_DO_REC (wctype)
    {
      TST_GET_ERRET (wctype);
      class = TST_INPUT (wctype).class;
      ret = wctype (class);

      if (debug_flg)
	{
	  fprintf (stderr, "tst_wctype : [ %d ] ret = %ld\n", rec + 1, ret);
	}

      TST_IF_RETURN (S_WCTYPE)
      {
	if (ret != 0)
	  {
	    Result (C_SUCCESS, S_WCTYPE, CASE_3, MS_PASSED);
	  }
	else
	  {
	    err_count++;
	    Result (C_FAILURE, S_WCTYPE, CASE_3,
		    "should return non-0, but returned 0");
	  }
      }
    }
  }

  return err_count;
}
