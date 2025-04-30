/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_isw-funcs.h
 *
 *	 ISW*:	int isw* (wint_t wc);
 */

#include <errno.h>
#include <stdlib.h>
#include <wctype.h>
#include "tst_types.h"
#include "tgn_locdef.h"

#define TST_ISW_LOC(FUNC, func) \
	TST_ISW## FUNC	  tst_isw## func ##_loc []

#define TST_ISW_REC(locale, func) \
	{  Tisw## func,	   TST_LOC_## locale  },

/*
 *  NOTE:
 *    Set ret_flg = 1, when a return value is expected to be 0 (FALSE).
 *    Set ret_flg = 0, when a return value is expected to be non-zero (TRUE).
 *
 *    Since the functions return *non*-zero value for TRUE, can't
 *    compare an actual return value with an expected return value.
 *    Set the ret_flg=0 for TRUE cases and the tst_isw*() will check
 *    the non-zero value.
 *
 *    { { WEOF }, { 0,1,0 } },
 *		      | |
 *		      | ret_val: an expected return value
 *		      ret_flg: if 1, compare an actual return value with the
 *			       ret_val; if 0, the test program
 *			       checks the actual return value.
 */
