/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	tst_funcs.h
 *
 *	 Definitions of macros
 */


#ifndef TST_FUNCS_H
#define TST_FUNCS_H

#define C_SUCCESS	   'S'	/* test case test passed		 */
#define C_FAILURE	   'F'	/* test case failed			 */
#define C_IGNORED	   'I'	/* test case/result ignored (not tested) */
#define C_INVALID	   'D'	/* test data may be wrong		 */
#define C_LOCALES	   'L'	/* can't set locale (skip)		 */


extern int result (FILE * fp, char res, const char *func, const char *loc,
		   int rec_no, int seq_num, int case_no, const char *msg);

#define Result(C, S, E, M) \
  result (fp, (C), (S), locale, rec+1, seq_num+1, (E), (M))

#define CASE_0	  0
#define CASE_1	  1
#define CASE_2	  2
#define CASE_3	  3
#define CASE_4	  4
#define CASE_5	  5
#define CASE_6	  6
#define CASE_7	  7
#define CASE_8	  8
#define CASE_9	  9

#define MS_PASSED "PASSED"
#define MS_SPACE  "	 "
#define MS_FAILED "	 "
#define MS_NOTEST "NOTEST"
#define MS_ABORTU "ABEND0"
#define MS_ABORT  "ABEND1"

#define MK_PASSED 0x00
#define MK_SPACE  0x01
#define MK_NOTEST 0x02
#define MK_ABORTU 0x04
#define MK_ABORT  0x08



/* ------------------ COMMON MACROS ------------------ */

#define TST_ABS(x)  (((x) > 0) ? (x) : -(x))

#define TMD_ERRET(_type_)   int	  err_val; \
			    int	  ret_flg; \
			    _type_ ret_val

#define TMD_RECHEAD(_FUNC_)				      \
									 \
			      typedef struct {				 \
				  TIN_##_FUNC_##_REC  input;		 \
				  TEX_##_FUNC_##_REC  expect;		 \
				  int is_last;				 \
			      }	  TST_##_FUNC_##_REC;			 \
			      typedef struct {				 \
				  TST_HEAD	      hd;		 \
				  TST_##_FUNC_##_REC  rec[ MAX_LOC_TEST ]; \
			      }	  TST_##_FUNC_

#define TST_FTYP(func)		tst_##func##_loc
#define TST_HEAD(func)		tst_##func##_loc[ loc ].hd
#define TST_INPUT(func)		tst_##func##_loc[ loc ].rec[ rec ].input
#define TST_EXPECT(func)	tst_##func##_loc[ loc ].rec[ rec ].expect
#define TST_INPUT_SEQ(func) \
	tst_##func##_loc[ loc ].rec[ rec ].input.seq[ seq_num ]
#define TST_EXPECT_SEQ(func) \
	tst_##func##_loc[ loc ].rec[ rec ].expect.seq[ seq_num ]
#define TST_IS_LAST(func) \
	tst_##func##_loc[ loc ].rec[ rec ].is_last


#define TST_DECL_VARS(_type_)				\
	int   loc, rec, err_count = 0;			\
	int   warn_count __attribute__ ((unused));	\
	int   seq_num = 0;				\
	const char *locale;				\
	int   err_exp, ret_flg;				\
	int errno_save = 0;				\
	_type_ ret_exp;					\
	_type_ ret

#define TST_DO_TEST(o_func) \
	for (loc = 0; strcmp (TST_HEAD (o_func).locale, TST_LOC_end); ++loc)


#define TST_HEAD_LOCALE(ofunc, s_func) \
  locale = TST_HEAD (ofunc).locale;					      \
  if (setlocale (LC_ALL, locale) == NULL)				      \
    {									      \
      fprintf (stderr, "Warning : can't set locale: %s\nskipping ...\n",      \
	       locale);							      \
      result (fp, C_LOCALES, s_func, locale, 0, 0, 0, "can't set locale");    \
      ++err_count;							      \
      continue;								      \
    }

#define TST_DO_REC(ofunc) \
	for (rec=0; !TST_IS_LAST (ofunc); ++rec)

#define TST_DO_SEQ(_count_) \
	for (seq_num=0; seq_num < _count_; seq_num++)

#define TST_GET_ERRET(_ofunc_)			\
	err_exp = TST_EXPECT (_ofunc_).err_val; \
	ret_flg = TST_EXPECT (_ofunc_).ret_flg; \
	ret_exp = TST_EXPECT (_ofunc_).ret_val

#define TST_GET_ERRET_SEQ(_ofunc_)		    \
	err_exp = TST_EXPECT_SEQ (_ofunc_).err_val; \
	ret_flg = TST_EXPECT_SEQ (_ofunc_).ret_flg; \
	ret_exp = TST_EXPECT_SEQ (_ofunc_).ret_val

#define TST_CLEAR_ERRNO \
	errno = 0

#define TST_SAVE_ERRNO \
	errno_save = errno

/* Test value of ret and of errno if it should have a value.  */
#define TST_IF_RETURN(_s_func_) \
  if (err_exp != 0)							      \
    {									      \
      if (errno_save == err_exp)					      \
	{								      \
	  result (fp, C_SUCCESS, _s_func_, locale, rec+1, seq_num+1, 1,	      \
		  MS_PASSED);						      \
	}								      \
      else								      \
	{								      \
	  err_count++;							      \
	  result (fp, C_FAILURE, _s_func_, locale, rec+1, seq_num+1, 1,	      \
		  "the value of errno is different from an expected value");  \
	}								      \
    }									      \
									      \
  if (ret_flg == 1)							      \
    {									      \
      if (ret == ret_exp)						      \
	{								      \
	  result (fp, C_SUCCESS, _s_func_, locale, rec+1, seq_num+1, 2,	      \
		  MS_PASSED);						      \
	}								      \
      else								      \
	{								      \
	  err_count++;							      \
	  result (fp, C_FAILURE, _s_func_, locale, rec+1, seq_num+1, 2,	      \
		  "the return value is different from an expected value");    \
	}								      \
    }									      \
  else

#define TEX_ERRET_REC(_type_)			\
	struct {				\
	    TMD_ERRET (_type_);			\
	}

#define TEX_ERRET_REC_SEQ(_type_, _count_)	\
	struct {				\
	    struct {				\
		TMD_ERRET (_type_);		\
	    } seq[ _count_ ];			\
	}



/* ------------------ FUNCTION: ISW*() ------------------- */

#define TST_ISW_STRUCT(_FUNC_, _func_)			\
	typedef						\
	struct {					\
	    wint_t   wc;				\
	} TIN_ISW##_FUNC_##_REC;			\
	typedef						\
	TEX_ERRET_REC (int)   TEX_ISW##_FUNC_##_REC;	\
	TMD_RECHEAD (ISW##_FUNC_)

#define TST_FUNC_ISW(_FUNC_, _func_) \
int									      \
tst_isw##_func_ (FILE *fp, int debug_flg)				      \
{									      \
  TST_DECL_VARS(int);							      \
  wint_t wc;								      \
  TST_DO_TEST (isw##_func_)						      \
    {									      \
      TST_HEAD_LOCALE (isw##_func_, S_ISW##_FUNC_);			      \
      TST_DO_REC(isw##_func_)						      \
	{								      \
	  TST_GET_ERRET (isw##_func_);					      \
	  wc = TST_INPUT (isw##_func_).wc;				      \
	  ret = isw##_func_ (wc);					      \
	  if (debug_flg)						      \
	    {								      \
	      fprintf (stdout, "isw*() [ %s : %d ] ret = %d\n", locale,	      \
		       rec+1, ret);					      \
	    }								      \
									      \
	  TST_IF_RETURN (S_ISW##_FUNC_)					      \
	    {								      \
	      if (ret != 0)						      \
		{							      \
		  result (fp, C_SUCCESS, S_ISW##_FUNC_, locale, rec+1,	      \
			  seq_num+1, 3, MS_PASSED);			      \
		}							      \
	      else							      \
		{							      \
		  err_count++;						      \
		  result (fp, C_FAILURE, S_ISW##_FUNC_, locale, rec+1,	      \
			  seq_num+1, 3,					      \
			  "the function returned 0, but should be non-zero"); \
		}							      \
	    }								      \
	}								      \
    }									      \
									      \
  return err_count;							      \
}



/* ------------------ FUNCTION: TOW*() ------------------ */

#define TST_TOW_STRUCT(_FUNC_, _func_)			\
	typedef						\
	struct {					\
	    wint_t   wc;				\
	} TIN_TOW##_FUNC_##_REC;			\
	typedef						\
	TEX_ERRET_REC (wint_t)	TEX_TOW##_FUNC_##_REC;	\
	TMD_RECHEAD (TOW##_FUNC_)

#define TST_FUNC_TOW(_FUNC_, _func_)					\
int									\
tst_tow##_func_ (FILE *fp, int debug_flg)				\
{									\
  TST_DECL_VARS (wint_t);						\
  wint_t wc;								\
  TST_DO_TEST (tow##_func_)						\
    {									\
      TST_HEAD_LOCALE (tow##_func_, S_TOW##_FUNC_);			\
      TST_DO_REC (tow##_func_)						\
	{								\
	  TST_GET_ERRET (tow##_func_);					\
	  wc = TST_INPUT (tow##_func_).wc;				\
	  ret = tow##_func_ (wc);					\
	  if (debug_flg)						\
	    {								\
	      fprintf (stdout, "tow*() [ %s : %d ] ret = 0x%x\n",	\
		       locale, rec+1, ret);				\
	    }								\
									\
	  TST_IF_RETURN (S_TOW##_FUNC_) { };				\
	}								\
    }									\
									\
  return err_count;							\
}


#endif /* TST_FUNCS_H */
