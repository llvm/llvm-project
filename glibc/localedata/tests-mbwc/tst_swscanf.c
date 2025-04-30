/*
  SWSCANF: int swscanf (const wchar_t *ws, const wchar_t *fmt, ...);
*/

#define TST_FUNCTION swscanf

#include "tsp_common.c"
#include "dat_swscanf.c"

int
tst_swscanf (FILE * fp, int debug_flg)
{
  TST_DECL_VARS (int);
  wchar_t *ws;
  wchar_t *fmt;
  int val_int1;
  unsigned val_int2;
  float val_flt;
  char val_c;
  char val_s[MBSSIZE * 3];
  wchar_t val_S[WCSSIZE * 3], *exp_S;
  int i;

  TST_DO_TEST (swscanf)
  {
    TST_HEAD_LOCALE (swscanf, S_SWSCANF);
    TST_DO_REC (swscanf)
    {
      TST_GET_ERRET (swscanf);
      ws = TST_INPUT (swscanf).ws;
      fmt = TST_INPUT (swscanf).fmt;
      val_int1 = val_int2 = val_flt = val_c = 0;
      memset (val_s, 0, sizeof (val_s));
      memset (val_S, 0, sizeof (val_S));

      TST_CLEAR_ERRNO;

      if (TST_INPUT (swscanf).wch)
	{
	  ret = swscanf (ws, fmt, val_S);
	}
      else
	{
	  ret =
	    swscanf (ws, fmt, &val_int1, &val_int2, &val_flt, &val_c, val_s);
	}

      TST_SAVE_ERRNO;

      if (debug_flg)
	{			/* seems fprintf doesn't update errno */
	  fprintf (stdout, "swscanf() [ %s : %d ] ret = %d\n", locale,
		   rec + 1, ret);
	  fprintf (stdout, "			    errno   = %d\n",
		   errno_save);
	  fprintf (stdout, "			    collate = %s\n",
		   (setlocale (LC_COLLATE, NULL)) ? setlocale (LC_COLLATE,
							       NULL) : "");

	  if (TST_INPUT (swscanf).wch)
	    {
	      fprintf (stdout, "			val_S[ 0 ] = 0x%lx\n",
		       (unsigned long int) val_S[0]);
	    }
	  else
	    {
	      fprintf (stdout, "			val_int1   = %d\n",
		       val_int1);
	      fprintf (stdout, "			val_int2   = %d\n",
		       val_int2);
	      fprintf (stdout, "			val_flt	   = %f\n",
		       val_flt);
	      fprintf (stdout, "			val_c	   = %c\n",
		       val_c);
	      fprintf (stdout, "			val_s	   = %s\n",
		       val_s);
	    }
	}

      TST_IF_RETURN (S_SWSCANF)
      {
      };

      if (errno == 0 && TST_INPUT (swscanf).wch)
	{
	  for (exp_S = TST_EXPECT (swscanf).val_S, i = 0; i < WCSSIZE * 3;
	       i++)
	    {
	      if (val_S[i] == L'\0' || exp_S[i] == L'\0')
		{
		  if (val_S[i] != exp_S[i] && TST_INPUT (swscanf).wch == 'C')
		    {
		      err_count++;
		      Result (C_FAILURE, S_SWSCANF, CASE_4,
			      "the converted wide-char string is different"
			      " from an expected value.");
		    }
		  break;
		}

	      if (val_S[i] != exp_S[i])
		{
		  err_count++;
		  Result (C_FAILURE, S_SWSCANF, CASE_4,
			  "the converted wide-char string is different from"
			  " an expected value.");
		  break;
		}
	      else
		{
		  Result (C_SUCCESS, S_SWSCANF, CASE_4, MS_PASSED);
		}
	    }
	}

      if (errno == 0 && !TST_INPUT (swscanf).wch)
	{
	  if (val_int1 != TST_EXPECT (swscanf).val_int
	      || val_int2 != TST_EXPECT (swscanf).val_uns
	      || val_flt != TST_EXPECT (swscanf).val_flt
	      || val_c != TST_EXPECT (swscanf).val_c
	      || strcmp (val_s, TST_EXPECT (swscanf).val_s))
	    {
	      err_count++;
	      Result (C_FAILURE, S_SWSCANF, CASE_3,
		      "the converted values are different from expected values.");
	    }
	  else
	    {
	      Result (C_SUCCESS, S_SWSCANF, CASE_3, MS_PASSED);
	    }
	}
    }
  }

  return err_count;
}
