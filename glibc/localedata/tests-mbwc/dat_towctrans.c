/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_towctrans.c
 *
 *	 TOWCTRANS:  wint_t towctrans (wint_t wc, wctrans_t charclass);
 */

#include <errno.h>
#include <stdlib.h>
#include <wctype.h>
#include "tst_types.h"
#include "tgn_locdef.h"

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
 *  { { WEOF }, { 0,0,1,0 } },
 *		      | |
 *		      | ret_val: an expected return value
 *		      ret_flg: if 1, compare an actual return value with the
 *			       ret_val; if 0, the test program checks
 *			       the actual return value.
 *
 *    CAUTION: if a charclass is invalid, the test function gives
 *    towctrans() an invalid wctrans object instead of a return value
 *    from wctrans() which is supposed to be 0.
 */

TST_TOWCTRANS tst_towctrans_loc [] = {
  {
    { Ttowctrans, TST_LOC_C },
    {
      {	 { 0x0010, "xxxxxxx" }, { 0,     1,0x0010 }  },
      {	 { 0x007F, "tolower" }, { 0,	   1,0x007F }  },
      {	 { 0x0061, "toupper" }, { 0,	   1,0x0041 }  },
      {	 { 0x0041, "tolower" }, { 0,	   1,0x0061 }  },
      { .is_last = 1 }
    }
  },
  {
    { Ttowctrans, TST_LOC_de },
    {
      {	 { 0x0010, "tojkata" }, { 0,     1,0x0010 }  },
      {	 { 0x0080, "tolower" }, { 0,	   1,0x0080 }  },
      {	 { 0x00EC, "toupper" }, { 0,	   1,0x00CC }  },
      {	 { 0x00CC, "tolower" }, { 0,	   1,0x00EC }  },
      { .is_last = 1 }
    }
  },
  {
    { Ttowctrans, TST_LOC_enUS },
    {
      {	 { 0x0010, "xxxxxxx" }, { 0,     1,0x0010 }  },
      {	 { 0x007F, "tolower" }, { 0,	   1,0x007F }  },
      {	 { 0x0061, "toupper" }, { 0,	   1,0x0041 }  },
      {	 { 0x0041, "tolower" }, { 0,	   1,0x0061 }  },
      { .is_last = 1 }
    }
  },
  {
    { Ttowctrans, TST_LOC_eucJP },
    {
      {	 { 0xFF21, "tolower" }, { 0,	   1,0xFF41 }  },
      {	 { 0xFF41, "toupper" }, { 0,	   1,0xFF21 }  },
      {	 { 0x30A1, "tojhira" }, { 0,	   1,0x3041 }  },
      {	 { 0x3041, "tojkata" }, { 0,	   1,0x30A1 }  },
      { .is_last = 1 }
    }
  },
  {
    { Ttowctrans, TST_LOC_end }
  }
};
