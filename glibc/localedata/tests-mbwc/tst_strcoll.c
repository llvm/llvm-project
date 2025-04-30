/*
  STRCOLL: int strcoll (const char *s1, const char *s2)
*/

#define TST_FUNCTION strcoll

#include "tsp_common.c"
#include "dat_strcoll.c"

int
tst_strcoll (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  const char *s1, *s2;

  TST_DO_TEST (strcoll)
  {
    TST_HEAD_LOCALE (strcoll, S_STRCOLL);
    TST_DO_REC (strcoll)
    {
      TST_GET_ERRET (strcoll);
      s1 = TST_INPUT (strcoll).s1;
      s2 = TST_INPUT (strcoll).s2;

      TST_CLEAR_ERRNO;
      ret = strcoll (s1, s2);
      TST_SAVE_ERRNO;

      if (debug_flg)
	{
	  fprintf (stdout, "strcoll() [ %s : %d ] ret = %d\n", locale,
		   rec + 1, ret);
	  fprintf (stdout, "			    errno = %d\n",
		   errno_save);
	  fprintf (stdout, "			    LC_COLLATE = %s\n",
		   (setlocale (LC_COLLATE, NULL)) ? setlocale (LC_COLLATE,
							       NULL) : "");
	}

      TST_IF_RETURN (S_STRCOLL)
      {
	if (ret_exp == +1)
	  {
	    if (ret > 0)
	      {
		Result (C_SUCCESS, S_STRCOLL, CASE_3, MS_PASSED);
	      }
	    else
	      {
		err_count++;
		Result (C_FAILURE, S_STRCOLL, CASE_3,
			"the return value should be greater than 0,"
			" but is not ...");
	      }
	  }
	else if (ret_exp == -1)
	  {
	    if (ret < 0)
	      {
		Result (C_SUCCESS, S_STRCOLL, CASE_3, MS_PASSED);
	      }
	    else
	      {
		err_count++;
		Result (C_FAILURE, S_STRCOLL, CASE_3,
			"the return value should less than 0, but not ...");
	      }
	  }
	else if (ret_exp != 0)
	  {
	    if (debug_flg)
	      {
		fprintf (stderr, "*** Warning *** : tst_strcoll : "
			 "(check the test data); should set ret_flg=1"
			 " to check a return value");
	      }

	    warn_count++;
	    Result (C_INVALID, S_WCSCHR, CASE_3, "(check the test data); "
		    "should set ret_flg=1 to check a return value");
	  }
      }
    }
  }

  return err_count;
}
