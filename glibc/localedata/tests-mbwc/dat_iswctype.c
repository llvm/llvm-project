/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_iswctype.c
 *
 *	 ISWCTYPE:  int iswctype( wint_t wc, wctype_t charclass );
 */

#include <errno.h>
#include <stdlib.h>
#include <wctype.h>
#include "tst_types.h"
#include "tgn_locdef.h"

/*
 *  NOTE:
 *   Set ret_flg = 1, when a return value is expected to be 0 (FALSE).
 *   Set ret_flg = 0, when a return value is expected to be non-zero (TRUE).
 *
 *   Since the functions return *non*-zero value for TRUE, can't
 *   compare an actual return value with an expected return value.
 *   Set the ret_flg=0 for TRUE cases and the tst_isw*() will check
 *   the non-zero value.
 *
 * { { WEOF }, { 0,1,0 } },
 *		   | |
 *		   | ret_val: an expected return value
 *		   ret_flg: if 1, compare an actual return value with the
 *			    ret_val; if 0, the test program checks
 *			    the actual return value.
 */

TST_ISWCTYPE tst_iswctype_loc [] = {
  {
    { Tiswctype, TST_LOC_de },
    {
      {	 { 0x009F, "alnum"  }, { 0,1,0 }  },	   /* CTRL     */
      {	 { 0x00A0, "alnum"  }, { 0,1,0 }  },	   /* NB SPACE */
      {	 { 0x00A1, "alnum"  }, { 0,1,0 }  },	   /* UD !     */
      {	 { 0x00B1, "alnum"  }, { 0,1,0 }  },	   /* +- sign  */
      {	 { 0x00B3, "alnum"  }, { 0,1,0 }  },	   /* SUP 3    */
      {	 { 0x00B4, "alnum"  }, { 0,1,0 }  },	   /* ACUTE    */
      {	 { 0x00BB, "alnum"  }, { 0,1,0 }  },	   /* >>       */
      {	 { 0x00BE, "alnum"  }, { 0,1,0 }  },	   /* 3/4      */
      {	 { 0x00BF, "alnum"  }, { 0,1,0 }  },	   /* UD ?     */
      {	 { 0x00C0, "alnum"  }, { 0,0,0 }  },	   /* A Grave  */
      {	 { 0x00D6, "alnum"  }, { 0,0,0 }  },	   /* O dia    */
      {	 { 0x00D7, "alnum"  }, { 0,1,0 }  },	   /* multipl. */
      {	 { 0x00D8, "alnum"  }, { 0,0,0 }  },	   /* O stroke */
      {	 { 0x00DF, "alnum"  }, { 0,0,0 }  },	   /* small Sh */
      {	 { 0x00E0, "alnum"  }, { 0,0,0 }  },	   /* a grave  */
      {	 { 0x00F6, "alnum"  }, { 0,0,0 }  },	   /* o dia    */
      {	 { 0x00F7, "alnum"  }, { 0,1,0 }  },	   /* division */
      {	 { 0x00F8, "alnum"  }, { 0,0,0 }  },	   /* o stroke */
      {	 { 0x00FF, "alnum"  }, { 0,0,0 }  },	   /* y dia    */
      {	 { 0x0080, "alpha"  }, { 0,1,0 }  },	   /* CTRL     */
      {	 { 0x00A0, "alpha"  }, { 0,1,0 }  },	   /* NB SPACE */
      {	 { 0x00A1, "alpha"  }, { 0,1,0 }  },	   /* UD !     */
      {	 { 0x00B1, "alpha"  }, { 0,1,0 }  },	   /* +- sign  */
      {	 { 0x00B4, "alpha"  }, { 0,1,0 }  },	   /* ACUTE    */
      {	 { 0x00B8, "alpha"  }, { 0,1,0 }  },	   /* CEDILLA  */
      {	 { 0x00B9, "alpha"  }, { 0,1,0 }  },	   /* SUP 1    */
      {	 { 0x00BB, "alpha"  }, { 0,1,0 }  },	   /* >>       */
      {	 { 0x00BE, "alpha"  }, { 0,1,0 }  },	   /* 3/4      */
      {	 { 0x00BF, "alpha"  }, { 0,1,0 }  },	   /* UD ?     */
      {	 { 0x00C0, "alpha"  }, { 0,0,0 }  },	   /* A Grave  */
      {	 { 0x00D6, "alpha"  }, { 0,0,0 }  },	   /* O dia    */
      {	 { 0x00D7, "alpha"  }, { 0,1,0 }  },	   /* multipl. */
      {	 { 0x00D8, "alpha"  }, { 0,0,0 }  },	   /* O stroke */
      {	 { 0x00DF, "alpha"  }, { 0,0,0 }  },	   /* small Sh */
      {	 { 0x00E0, "alpha"  }, { 0,0,0 }  },	   /* a grave  */
      {	 { 0x00F6, "alpha"  }, { 0,0,0 }  },	   /* o dia    */
      {	 { 0x00F7, "alpha"  }, { 0,1,0 }  },	   /* division */
      {	 { 0x00F8, "alpha"  }, { 0,0,0 }  },	   /* o stroke */
      {	 { 0x00FF, "alpha"  }, { 0,0,0 }  },	   /* y dia    */
      {	 { 0x0080, "cntrl"  }, { 0,0,0 }  },	   /* CTRL     */
      {	 { 0x009F, "cntrl"  }, { 0,0,0 }  },	   /* CTRL     */
      {	 { 0x00A0, "cntrl"  }, { 0,1,0 }  },	   /* NB SPACE */
      {	 { 0x00F6, "cntrl"  }, { 0,1,0 }  },	   /* o dia    */
      {	 { 0x00FF, "cntrl"  }, { 0,1,0 }  },	   /* y dia    */
      {	 { 0x00B9, "digit"  }, { 0,1,0 }  },	   /* SUP 1    */
      {	 { 0x00BE, "digit"  }, { 0,1,0 }  },	   /* 3/4      */
      {	 { 0x009F, "graph"  }, { 0,1,0 }  },	   /* CTRL     */
      {	 { 0x00A0, "graph"  }, { 0,0,0 }  },	   /* NB SPACE */
      {	 { 0x00A1, "graph"  }, { 0,0,0 }  },	   /* UD !     */
      {	 { 0x00B1, "graph"  }, { 0,0,0 }  },	   /* +- sign  */
      {	 { 0x00B3, "graph"  }, { 0,0,0 }  },	   /* SUP 3    */
      {	 { 0x00B4, "graph"  }, { 0,0,0 }  },	   /* ACUTE    */
      {	 { 0x00BB, "graph"  }, { 0,0,0 }  },	   /* >>       */
      {	 { 0x00BE, "graph"  }, { 0,0,0 }  },	   /* 3/4      */
      {	 { 0x00C0, "graph"  }, { 0,0,0 }  },	   /* A Grave  */
      {	 { 0x00D6, "graph"  }, { 0,0,0 }  },	   /* O dia    */
      {	 { 0x00D7, "graph"  }, { 0,0,0 }  },	   /* multipl. */
      {	 { 0x00D8, "graph"  }, { 0,0,0 }  },	   /* O stroke */
      {	 { 0x00DF, "graph"  }, { 0,0,0 }  },	   /* small Sh */
      {	 { 0x00F7, "graph"  }, { 0,0,0 }  },	   /* division */
      {	 { 0x00F8, "graph"  }, { 0,0,0 }  },	   /* o stroke */
      {	 { 0x00FF, "graph"  }, { 0,0,0 }  },	   /* y dia    */
      {	 { 0x009F, "print"  }, { 0,1,0 }  },	   /* CTRL     */
      {	 { 0x00A0, "print"  }, { 0,0,0 }  },	   /* NB SPACE */
      {	 { 0x00A1, "print"  }, { 0,0,0 }  },	   /* UD !     */
      {	 { 0x00B1, "print"  }, { 0,0,0 }  },	   /* +- sign  */
      {	 { 0x00B4, "print"  }, { 0,0,0 }  },	   /* ACUTE    */
      {	 { 0x00B8, "print"  }, { 0,0,0 }  },	   /* CEDILLA  */
      {	 { 0x00B9, "print"  }, { 0,0,0 }  },	   /* SUP 1    */
      {	 { 0x00BB, "print"  }, { 0,0,0 }  },	   /* >>       */
      {	 { 0x00BE, "print"  }, { 0,0,0 }  },	   /* 3/4      */
      {	 { 0x00C0, "print"  }, { 0,0,0 }  },	   /* A Grave  */
      {	 { 0x00DF, "print"  }, { 0,0,0 }  },	   /* small Sh */
      {	 { 0x00F6, "print"  }, { 0,0,0 }  },	   /* o dia    */
      {	 { 0x00F7, "print"  }, { 0,0,0 }  },	   /* division */
      {	 { 0x00F8, "print"  }, { 0,0,0 }  },	   /* o stroke */
      {	 { 0x00FF, "print"  }, { 0,0,0 }  },	   /* y dia    */
      {	 { 0x009F, "punct"  }, { 0,1,0 }  },	   /* CTRL     */
      {	 { 0x00A0, "punct"  }, { 0,0,0 }  },	   /* NB SPACE */
      {	 { 0x00A1, "punct"  }, { 0,0,0 }  },	   /* UD !     */
      {	 { 0x00B0, "punct"  }, { 0,0,0 }  },	   /* Degree   */
      {	 { 0x00B1, "punct"  }, { 0,0,0 }  },	   /* +- sign  */
      {	 { 0x00B2, "punct"  }, { 0,0,0 }  },	   /* SUP 2    */
      {	 { 0x00B3, "punct"  }, { 0,0,0 }  },	   /* SUP 3    */
      {	 { 0x00B4, "punct"  }, { 0,0,0 }  },	   /* ACUTE    */
      {	 { 0x00B8, "punct"  }, { 0,0,0 }  },	   /* CEDILLA  */
      {	 { 0x00B9, "punct"  }, { 0,0,0 }  },	   /* SUP 1    */
      {	 { 0x00BB, "punct"  }, { 0,0,0 }  },	   /* >>       */
      {	 { 0x00BC, "punct"  }, { 0,0,0 }  },	   /* 1/4      */
      {	 { 0x00BD, "punct"  }, { 0,0,0 }  },	   /* 1/2      */
      {	 { 0x00BE, "punct"  }, { 0,0,0 }  },	   /* 3/4      */
      {	 { 0x00BF, "punct"  }, { 0,0,0 }  },	   /* UD ?     */
      {	 { 0x00C0, "punct"  }, { 0,1,0 }  },	   /* A Grave  */
      {	 { 0x00D7, "punct"  }, { 0,0,0 }  },	   /* multipl. */
      {	 { 0x00DF, "punct"  }, { 0,1,0 }  },	   /* small Sh */
      {	 { 0x00F6, "punct"  }, { 0,1,0 }  },	   /* o dia    */
      {	 { 0x00F7, "punct"  }, { 0,0,0 }  },	   /* division */
      {	 { 0x00FF, "punct"  }, { 0,1,0 }  },	   /* y dia    */
      {	 { 0x009F, "space"  }, { 0,1,0 }  },	   /* CTRL     */
      {	 { 0x00A0, "space"  }, { 0,1,0 }  },	   /* NB SPACE */
      {	 { 0x00A1, "space"  }, { 0,1,0 }  },	   /* UD !     */
      {	 { 0x00B1, "space"  }, { 0,1,0 }  },	   /* +- sign  */
      {	 { 0x00F8, "space"  }, { 0,1,0 }  },	   /* o stroke */
      {	 { 0x00B3, "lower"  }, { 0,1,0 }  },	   /* SUP 3    */
      {	 { 0x00B8, "lower"  }, { 0,1,0 }  },	   /* CEDILLA  */
      {	 { 0x00BE, "lower"  }, { 0,1,0 }  },	   /* 3/4      */
      {	 { 0x00C0, "lower"  }, { 0,1,0 }  },	   /* A Grave  */
      {	 { 0x00D6, "lower"  }, { 0,1,0 }  },	   /* O dia    */
      {	 { 0x00D8, "lower"  }, { 0,1,0 }  },	   /* O stroke */
      {	 { 0x00DF, "lower"  }, { 0,0,0 }  },	   /* small Sh */
      {	 { 0x00E0, "lower"  }, { 0,0,0 }  },	   /* a grave  */
      {	 { 0x00F6, "lower"  }, { 0,0,0 }  },	   /* o dia    */
      {	 { 0x00F7, "lower"  }, { 0,1,0 }  },	   /* division */
      {	 { 0x00F8, "lower"  }, { 0,0,0 }  },	   /* o stroke */
      {	 { 0x00FF, "lower"  }, { 0,0,0 }  },	   /* y dia    */
      {	 { 0x00B4, "upper"  }, { 0,1,0 }  },	   /* ACUTE    */
      {	 { 0x00B8, "upper"  }, { 0,1,0 }  },	   /* CEDILLA  */
      {	 { 0x00B9, "upper"  }, { 0,1,0 }  },	   /* SUP 1    */
      {	 { 0x00BE, "upper"  }, { 0,1,0 }  },	   /* 3/4      */
      {	 { 0x00BF, "upper"  }, { 0,1,0 }  },	   /* UD ?     */
      {	 { 0x00C0, "upper"  }, { 0,0,0 }  },	   /* A Grave  */
      {	 { 0x00D6, "upper"  }, { 0,0,0 }  },	   /* O dia    */
      {	 { 0x00D7, "upper"  }, { 0,1,0 }  },	   /* multipl. */
      {	 { 0x00D8, "upper"  }, { 0,0,0 }  },	   /* O stroke */
      {	 { 0x00DF, "upper"  }, { 0,1,0 }  },	   /* small Sh */
      {	 { 0x00FF, "upper"  }, { 0,1,0 }  },	   /* y dia    */
      {	 { 0x00B9, "xdigit" }, { 0,1,0 }  },	   /* SUP 1    */
      {	 { 0x00BC, "xdigit" }, { 0,1,0 }  },	   /* 1/4      */
      { .is_last = 1 }
    }
  },
  {
    { Tiswctype, TST_LOC_enUS },
    {
      {	 { WEOF,   "alnum"  }, { 0,1,0 }  },
      {	 { 0x0000, "alnum"  }, { 0,1,0 }  },
      {	 { 0x001F, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0020, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0021, "alnum"  }, { 0,1,0 }  },
      {	 { 0x002F, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0030, "alnum"  }, { 0,0,0 }  },
      {	 { 0x0039, "alnum"  }, { 0,0,0 }  },
      {	 { 0x003A, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0040, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0041, "alnum"  }, { 0,0,0 }  },
      {	 { 0x005A, "alnum"  }, { 0,0,0 }  },
      {	 { 0x005B, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0060, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0061, "alnum"  }, { 0,0,0 }  },
      {	 { 0x007A, "alnum"  }, { 0,0,0 }  },
      {	 { 0x007B, "alnum"  }, { 0,1,0 }  },
      {	 { 0x007E, "alnum"  }, { 0,1,0 }  },
      {	 { 0x007F, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0080, "alnum"  }, { 0,1,0 }  },
      {	 { 0x0000, "alpha"  }, { 0,1,0 }  },
      {	 { 0x001F, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0020, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0021, "alpha"  }, { 0,1,0 }  },
      {	 { 0x002F, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0030, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0039, "alpha"  }, { 0,1,0 }  },
      {	 { 0x003A, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0040, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0041, "alpha"  }, { 0,0,0 }  },
      {	 { 0x005A, "alpha"  }, { 0,0,0 }  },
      {	 { 0x005B, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0060, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0061, "alpha"  }, { 0,0,0 }  },
      {	 { 0x007A, "alpha"  }, { 0,0,0 }  },
      {	 { 0x007B, "alpha"  }, { 0,1,0 }  },
      {	 { 0x007E, "alpha"  }, { 0,1,0 }  },
      {	 { 0x007F, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0080, "alpha"  }, { 0,1,0 }  },
      {	 { 0x0009, "blank"  }, { 0,0,0 }  },
      {	 { 0x000B, "blank"  }, { 0,1,0 }  },
      {	 { 0x0020, "blank"  }, { 0,0,0 }  },
      {	 { 0x0000, "cntrl"  }, { 0,0,0 }  },
      {	 { 0x001F, "cntrl"  }, { 0,0,0 }  },
      {	 { 0x0020, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x0021, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x002F, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x0030, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x0039, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x003A, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x0040, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x0041, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x005A, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x005B, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x0060, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x0061, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x007A, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x007B, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x007E, "cntrl"  }, { 0,1,0 }  },
      {	 { 0x007F, "cntrl"  }, { 0,0,0 }  },
      {	 { 0x0080, "cntrl"  }, { 0,0,0 }  },
      {	 { 0x0000, "digit"  }, { 0,1,0 }  },
      {	 { 0x001F, "digit"  }, { 0,1,0 }  },
      {	 { 0x0020, "digit"  }, { 0,1,0 }  },
      {	 { 0x0021, "digit"  }, { 0,1,0 }  },
      {	 { 0x002F, "digit"  }, { 0,1,0 }  },
      {	 { 0x0030, "digit"  }, { 0,0,0 }  },
      {	 { 0x0039, "digit"  }, { 0,0,0 }  },
      {	 { 0x003A, "digit"  }, { 0,1,0 }  },
      {	 { 0x0040, "digit"  }, { 0,1,0 }  },
      {	 { 0x0041, "digit"  }, { 0,1,0 }  },
      {	 { 0x005A, "digit"  }, { 0,1,0 }  },
      {	 { 0x005B, "digit"  }, { 0,1,0 }  },
      {	 { 0x0060, "digit"  }, { 0,1,0 }  },
      {	 { 0x0061, "digit"  }, { 0,1,0 }  },
      {	 { 0x007A, "digit"  }, { 0,1,0 }  },
      {	 { 0x007B, "digit"  }, { 0,1,0 }  },
      {	 { 0x007E, "digit"  }, { 0,1,0 }  },
      {	 { 0x007F, "digit"  }, { 0,1,0 }  },
      {	 { 0x0080, "digit"  }, { 0,1,0 }  },
      {	 { 0x0000, "graph"  }, { 0,1,0 }  },
      {	 { 0x001F, "graph"  }, { 0,1,0 }  },
      {	 { 0x0020, "graph"  }, { 0,1,0 }  },
      {	 { 0x0021, "graph"  }, { 0,0,0 }  },
      {	 { 0x002F, "graph"  }, { 0,0,0 }  },
      {	 { 0x0030, "graph"  }, { 0,0,0 }  },
      {	 { 0x0039, "graph"  }, { 0,0,0 }  },
      {	 { 0x003A, "graph"  }, { 0,0,0 }  },
      {	 { 0x0040, "graph"  }, { 0,0,0 }  },
      {	 { 0x0041, "graph"  }, { 0,0,0 }  },
      {	 { 0x005A, "graph"  }, { 0,0,0 }  },
      {	 { 0x005B, "graph"  }, { 0,0,0 }  },
      {	 { 0x0060, "graph"  }, { 0,0,0 }  },
      {	 { 0x0061, "graph"  }, { 0,0,0 }  },
      {	 { 0x007A, "graph"  }, { 0,0,0 }  },
      {	 { 0x007B, "graph"  }, { 0,0,0 }  },
      {	 { 0x007E, "graph"  }, { 0,0,0 }  },
      {	 { 0x007F, "graph"  }, { 0,1,0 }  },
      {	 { 0x0080, "graph"  }, { 0,1,0 }  },
      {	 { 0x0000, "print"  }, { 0,1,0 }  },
      {	 { 0x001F, "print"  }, { 0,1,0 }  },
      {	 { 0x0020, "print"  }, { 0,0,0 }  },
      {	 { 0x0021, "print"  }, { 0,0,0 }  },
      {	 { 0x002F, "print"  }, { 0,0,0 }  },
      {	 { 0x0030, "print"  }, { 0,0,0 }  },
      {	 { 0x0039, "print"  }, { 0,0,0 }  },
      {	 { 0x003A, "print"  }, { 0,0,0 }  },
      {	 { 0x0040, "print"  }, { 0,0,0 }  },
      {	 { 0x0041, "print"  }, { 0,0,0 }  },
      {	 { 0x005A, "print"  }, { 0,0,0 }  },
      {	 { 0x005B, "print"  }, { 0,0,0 }  },
      {	 { 0x0060, "print"  }, { 0,0,0 }  },
      {	 { 0x0061, "print"  }, { 0,0,0 }  },
      {	 { 0x007A, "print"  }, { 0,0,0 }  },
      {	 { 0x007B, "print"  }, { 0,0,0 }  },
      {	 { 0x007E, "print"  }, { 0,0,0 }  },
      {	 { 0x007F, "print"  }, { 0,1,0 }  },
      {	 { 0x0080, "print"  }, { 0,1,0 }  },
      {	 { 0x0000, "punct"  }, { 0,1,0 }  },
      {	 { 0x001F, "punct"  }, { 0,1,0 }  },
      {	 { 0x0020, "punct"  }, { 0,1,0 }  },
      {	 { 0x0021, "punct"  }, { 0,0,0 }  },
      {	 { 0x002F, "punct"  }, { 0,0,0 }  },
      {	 { 0x0030, "punct"  }, { 0,1,0 }  },
      {	 { 0x0039, "punct"  }, { 0,1,0 }  },
      {	 { 0x003A, "punct"  }, { 0,0,0 }  },
      {	 { 0x0040, "punct"  }, { 0,0,0 }  },
      {	 { 0x0041, "punct"  }, { 0,1,0 }  },
      {	 { 0x005A, "punct"  }, { 0,1,0 }  },
      {	 { 0x005B, "punct"  }, { 0,0,0 }  },
      {	 { 0x0060, "punct"  }, { 0,0,0 }  },
      {	 { 0x0061, "punct"  }, { 0,1,0 }  },
      {	 { 0x007A, "punct"  }, { 0,1,0 }  },
      {	 { 0x007B, "punct"  }, { 0,0,0 }  },
      {	 { 0x007E, "punct"  }, { 0,0,0 }  },
      {	 { 0x007F, "punct"  }, { 0,1,0 }  },
      {	 { 0x0080, "punct"  }, { 0,1,0 }  },
      {	 { 0x0000, "space"  }, { 0,1,0 }  },
      {	 { 0x001F, "space"  }, { 0,1,0 }  },
      {	 { 0x0020, "space"  }, { 0,0,0 }  },
      {	 { 0x0021, "space"  }, { 0,1,0 }  },
      {	 { 0x002F, "space"  }, { 0,1,0 }  },
      {	 { 0x007E, "space"  }, { 0,1,0 }  },
      {	 { 0x007F, "space"  }, { 0,1,0 }  },
      {	 { 0x0080, "space"  }, { 0,1,0 }  },
      {	 { 0x0000, "lower"  }, { 0,1,0 }  },
      {	 { 0x001F, "lower"  }, { 0,1,0 }  },
      {	 { 0x0020, "lower"  }, { 0,1,0 }  },
      {	 { 0x0021, "lower"  }, { 0,1,0 }  },
      {	 { 0x002F, "lower"  }, { 0,1,0 }  },
      {	 { 0x0030, "lower"  }, { 0,1,0 }  },
      {	 { 0x0039, "lower"  }, { 0,1,0 }  },
      {	 { 0x003A, "lower"  }, { 0,1,0 }  },
      {	 { 0x0040, "lower"  }, { 0,1,0 }  },
      {	 { 0x0041, "lower"  }, { 0,1,0 }  },
      {	 { 0x005A, "lower"  }, { 0,1,0 }  },
      {	 { 0x005B, "lower"  }, { 0,1,0 }  },
      {	 { 0x0060, "lower"  }, { 0,1,0 }  },
      {	 { 0x0061, "lower"  }, { 0,0,0 }  },
      {	 { 0x007A, "lower"  }, { 0,0,0 }  },
      {	 { 0x007B, "lower"  }, { 0,1,0 }  },
      {	 { 0x007E, "lower"  }, { 0,1,0 }  },
      {	 { 0x007F, "lower"  }, { 0,1,0 }  },
      {	 { 0x0080, "lower"  }, { 0,1,0 }  },
      {	 { 0x0000, "upper"  }, { 0,1,0 }  },
      {	 { 0x001F, "upper"  }, { 0,1,0 }  },
      {	 { 0x0020, "upper"  }, { 0,1,0 }  },
      {	 { 0x0021, "upper"  }, { 0,1,0 }  },
      {	 { 0x002F, "upper"  }, { 0,1,0 }  },
      {	 { 0x0030, "upper"  }, { 0,1,0 }  },
      {	 { 0x0039, "upper"  }, { 0,1,0 }  },
      {	 { 0x003A, "upper"  }, { 0,1,0 }  },
      {	 { 0x0040, "upper"  }, { 0,1,0 }  },
      {	 { 0x0041, "upper"  }, { 0,0,0 }  },
      {	 { 0x005A, "upper"  }, { 0,0,0 }  },
      {	 { 0x005B, "upper"  }, { 0,1,0 }  },
      {	 { 0x0060, "upper"  }, { 0,1,0 }  },
      {	 { 0x0061, "upper"  }, { 0,1,0 }  },
      {	 { 0x007A, "upper"  }, { 0,1,0 }  },
      {	 { 0x007B, "upper"  }, { 0,1,0 }  },
      {	 { 0x007E, "upper"  }, { 0,1,0 }  },
      {	 { 0x007F, "upper"  }, { 0,1,0 }  },
      {	 { 0x0080, "upper"  }, { 0,1,0 }  },
      {	 { 0x0000, "xdigit" }, { 0,1,0 }  },
      {	 { 0x001F, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0020, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0021, "xdigit" }, { 0,1,0 }  },
      {	 { 0x002F, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0030, "xdigit" }, { 0,0,0 }  },
      {	 { 0x0039, "xdigit" }, { 0,0,0 }  },
      {	 { 0x003A, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0040, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0041, "xdigit" }, { 0,0,0 }  },
      {	 { 0x005A, "xdigit" }, { 0,1,0 }  },
      {	 { 0x005B, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0060, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0061, "xdigit" }, { 0,0,0 }  },
      {	 { 0x007A, "xdigit" }, { 0,1,0 }  },
      {	 { 0x007B, "xdigit" }, { 0,1,0 }  },
      {	 { 0x007E, "xdigit" }, { 0,1,0 }  },
      {	 { 0x007F, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0080, "xdigit" }, { 0,1,0 }  },
      {	 { 0x0061, "xxxxxx" }, { 0,1,0 }  },
      { .is_last = 1 }
    }
  },
  {
    { Tiswctype, TST_LOC_eucJP },
    {
      {	 { 0x3029, "alnum"  }, { 0,0,0 }  },	   /* Hangzhou NUM9	 */
      {	 { 0xFE4F, "alnum"  }, { 0,1,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0xFF19, "alnum"  }, { 0,0,0 }  },	   /* FULL 9		 */
      {	 { 0xFF20, "alnum"  }, { 0,1,0 }  },	   /* FULL @		 */
      {	 { 0xFF3A, "alnum"  }, { 0,0,0 }  },	   /* FULL Z		 */
      {	 { 0xFF40, "alnum"  }, { 0,1,0 }  },	   /* FULL GRAVE ACC.	 */
      {	 { 0xFF5A, "alnum"  }, { 0,0,0 }  },	   /* FULL z		 */
      {	 { 0xFF71, "alnum"  }, { 0,0,0 }  },	   /* HALF KATA A	 */
      {	 { 0x3029, "alpha"  }, { 0,0,0 }  },	   /* Hangzhou NUM9	 */
      {	 { 0xFE4F, "alpha"  }, { 0,1,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0xFF19, "alpha"  }, { 0,0,0 }  },	   /* FULL 9		 */
      {	 { 0xFF20, "alpha"  }, { 0,1,0 }  },	   /* FULL @		 */
      {	 { 0xFF3A, "alpha"  }, { 0,0,0 }  },	   /* FULL Z		 */
      {	 { 0xFF40, "alpha"  }, { 0,1,0 }  },	   /* FULL GRAVE ACC.	 */
      {	 { 0xFF5A, "alpha"  }, { 0,0,0 }  },	   /* FULL z		 */
      {	 { 0xFF71, "alpha"  }, { 0,0,0 }  },	   /* HALF KATA A	 */
      {	 { 0x0080, "cntrl"  }, { 0,0,0 }  },	   /* CNTRL		 */
      {	 { 0x3000, "cntrl"  }, { 0,1,0 }  },	   /* IDEO. SPACE	 */
      {	 { 0x3029, "digit"  }, { 0,1,0 }  },	   /* Hangzhou NUM9	 */
      {	 { 0x32CB, "digit"  }, { 0,1,0 }  },	   /* IDEO.TEL.SYM.DEC12 */
      /* 21: */
      {	 { 0x33FE, "digit"  }, { 0,1,0 }  },	   /* CJK IDEO.TEL.31th	 */
      {	 { 0xFF19, "digit"  }, { 0,1,0 }  },	   /* FULL 9		 */
      {	 { 0x3000, "graph"  }, { 0,1,0 }  },	   /* IDEO. SPACE	 */
      {	 { 0x3020, "graph"  }, { 0,0,0 }  },	   /* POSTAL MARK FACE	 */
      {	 { 0x3029, "graph"  }, { 0,0,0 }  },	   /* Hangzhou NUM9	 */
      {	 { 0x302F, "graph"  }, { 0,0,0 }  },	   /* Diacritics(Hangul) */
      {	 { 0x3037, "graph"  }, { 0,0,0 }  },	   /* Separator Symbol	 */
      {	 { 0x303F, "graph"  }, { 0,0,0 }  },	   /* IDEO. HALF SPACE	 */
      /* 29: */
      {	 { 0x3041, "graph"  }, { 0,0,0 }  },	   /* HIRAGANA a	 */
      /* Non jis: */
      {	 { 0x3094, "graph"  }, { 0,0,0 }  },	   /* HIRAGANA u"	 */
      /* Non jis: */
      {	 { 0x3099, "graph"  }, { 0,0,0 }  },	   /* SOUND MARK	 */
      {	 { 0x309E, "graph"  }, { 0,0,0 }  },	   /* ITERATION MARK	 */
      /* 33: */
      {	 { 0x30A1, "graph"  }, { 0,0,0 }  },	   /* KATAKANA a	 */
      /* Non jis: */
      {	 { 0x30FA, "graph"  }, { 0,0,0 }  },	   /* KATAKANA wo"	 */
      {	 { 0x30FB, "graph"  }, { 0,0,0 }  },	   /* KATAKANA MID.DOT	 */
      {	 { 0x30FE, "graph"  }, { 0,0,0 }  },	   /* KATAKANA ITERATION */
      {	 { 0x3191, "graph"  }, { 0,0,0 }  },	   /* KANBUN REV.MARK	 */
      {	 { 0x3243, "graph"  }, { 0,0,0 }  },	   /* IDEO. MARK (reach) */
      {	 { 0x32CB, "graph"  }, { 0,0,0 }  },	   /* IDEO.TEL.SYM.DEC12 */
      {	 { 0x32FE, "graph"  }, { 0,0,0 }  },	   /* MARU KATAKANA wo	 */
      {	 { 0x33FE, "graph"  }, { 0,0,0 }  },	   /* CJK IDEO.TEL.31th	 */
      {	 { 0x4E00, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x4E05, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x4E06, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.NON-J */
      {	 { 0x4E07, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x4FFF, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x9000, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x9006, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x9007, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.NON-J */
      {	 { 0x9FA4, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.NON-J */
      /* 51 */
      {	 { 0x9FA5, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      /* Non jis: */
      {	 { 0xFE4F, "graph"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0xFF0F, "graph"  }, { 0,0,0 }  },	   /* FULL SLASH	 */
      {	 { 0xFF19, "graph"  }, { 0,0,0 }  },	   /* FULL 9		 */
      {	 { 0xFF20, "graph"  }, { 0,0,0 }  },	   /* FULL @		 */
      {	 { 0xFF3A, "graph"  }, { 0,0,0 }  },	   /* FULL Z		 */
      {	 { 0xFF40, "graph"  }, { 0,0,0 }  },	   /* FULL GRAVE ACC.	 */
      {	 { 0xFF5A, "graph"  }, { 0,0,0 }  },	   /* FULL z		 */
      {	 { 0xFF5E, "graph"  }, { 0,0,0 }  },	   /* FULL ~ (tilde)	 */
      {	 { 0xFF61, "graph"  }, { 0,0,0 }  },	   /* HALF IDEO.STOP. .	 */
      {	 { 0xFF65, "graph"  }, { 0,0,0 }  },	   /* HALF KATA MID.DOT	 */
      {	 { 0xFF66, "graph"  }, { 0,0,0 }  },	   /* HALF KATA WO	 */
      {	 { 0xFF6F, "graph"  }, { 0,0,0 }  },	   /* HALF KATA tu	 */
      {	 { 0xFF70, "graph"  }, { 0,0,0 }  },	   /* HALF KATA PL -	 */
      {	 { 0xFF71, "graph"  }, { 0,0,0 }  },	   /* HALF KATA A	 */
      {	 { 0xFF9E, "graph"  }, { 0,0,0 }  },	   /* HALF KATA MI	 */
      {	 { 0x3000, "print"  }, { 0,0,0 }  },	   /* IDEO. SPACE	 */
      {	 { 0x3020, "print"  }, { 0,0,0 }  },	   /* POSTAL MARK FACE	 */
      {	 { 0x3029, "print"  }, { 0,0,0 }  },	   /* Hangzhou NUM9	 */
      {	 { 0x302F, "print"  }, { 0,0,0 }  },	   /* Diacritics(Hangul) */
      {	 { 0x3037, "print"  }, { 0,0,0 }  },	   /* Separator Symbol	 */
      {	 { 0x4E00, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x4E05, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x4E06, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.NON-J */
      {	 { 0x4E07, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x4FFF, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x9000, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x9006, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x9007, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.NON-J */
      {	 { 0x9FA4, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.NON-J */
      /* 81: */
      {	 { 0x9FA5, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      /* Non jis: */
      {	 { 0xFE4F, "print"  }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0x3000, "punct"  }, { 0,1,0 }  },	   /* IDEO. SPACE	 */
      {	 { 0x3020, "punct"  }, { 0,0,0 }  },	   /* POSTAL MARK FACE	 */
      {	 { 0x302F, "punct"  }, { 0,0,0 }  },	   /* Diacritics(Hangul) */
      {	 { 0x3037, "punct"  }, { 0,0,0 }  },	   /* FEED Separator	 */
      {	 { 0x303F, "punct"  }, { 0,0,0 }  },	   /* IDEO. HALF SPACE	 */
      {	 { 0x3041, "punct"  }, { 0,1,0 }  },	   /* HIRAGANA a	 */
      {	 { 0x3094, "punct"  }, { 0,1,0 }  },	   /* HIRAGANA u"	 */
      /* 90: */
      {	 { 0x3099, "punct"  }, { 0,0,0 }  },	   /* SOUND MARK	 */
      {	 { 0x309E, "punct"  }, { 0,1,0 }  },	   /* ITERATION MARK	 */
      {	 { 0x30A1, "punct"  }, { 0,1,0 }  },	   /* KATAKANA a	 */
      {	 { 0x30FA, "punct"  }, { 0,1,0 }  },	   /* KATAKANA wo"	 */
      {	 { 0x30FB, "punct"  }, { 0,0,0 }  },	   /* KATAKANA MID.DOT	 */
      /* 95: */
      {	 { 0x30FE, "punct"  }, { 0,1,0 }  },	   /* KATAKANA ITERATION */
      {	 { 0x3191, "punct"  }, { 0,0,0 }  },	   /* KANBUN REV.MARK	 */
      {	 { 0x3243, "punct"  }, { 0,0,0 }  },	   /* IDEO. MARK (reach) */
      {	 { 0x32CB, "punct"  }, { 0,0,0 }  },	   /* IDEO.TEL.SYM.DEC12 */
      {	 { 0x32FE, "punct"  }, { 0,0,0 }  },	   /* MARU KATAKANA wo	 */
      {	 { 0x33FE, "punct"  }, { 0,0,0 }  },	   /* CJK IDEO.TEL.31th	 */
      {	 { 0x9007, "punct"  }, { 0,1,0 }  },	   /* CJK UNI.IDEO.NON-J */
      {	 { 0x9FA4, "punct"  }, { 0,1,0 }  },	   /* CJK UNI.IDEO.NON-J */
      {	 { 0x9FA5, "punct"  }, { 0,1,0 }  },	   /* CJK UNI.IDEO.	 */
      {	 { 0xFF0F, "punct"  }, { 0,0,0 }  },	   /* FULL SLASH	 */
      /* 105: */
      {	 { 0xFF19, "punct"  }, { 0,1,0 }  },	   /* FULL 9		 */
      {	 { 0xFF20, "punct"  }, { 0,0,0 }  },	   /* FULL @		 */
      {	 { 0xFF3A, "punct"  }, { 0,1,0 }  },	   /* FULL Z		 */
      {	 { 0xFF40, "punct"  }, { 0,0,0 }  },	   /* FULL GRAVE ACC.	 */
      {	 { 0xFF5A, "punct"  }, { 0,1,0 }  },	   /* FULL z		 */
      {	 { 0xFF5E, "punct"  }, { 0,0,0 }  },	   /* FULL ~ (tilde)	 */
      {	 { 0xFF61, "punct"  }, { 0,0,0 }  },	   /* HALF IDEO.STOP. .	 */
      {	 { 0xFF65, "punct"  }, { 0,0,0 }  },	   /* HALF KATA MID.DOT	 */
      {	 { 0xFF70, "punct"  }, { 0,1,0 }  },	   /* HALF KATA PL -	 */
      {	 { 0xFF9E, "punct"  }, { 0,1,0 }  },	   /* HALF KATA MI	 */
      /* 115: */
      {	 { 0x3000, "space"  }, { 0,0,0 }  },	   /* IDEO. SPACE	 */
      {	 { 0x303F, "space"  }, { 0,1,0 }  },	   /* IDEO. HALF SPACE	 */
      {	 { 0x3041, "lower"  }, { 0,1,0 }  },	   /* HIRAGANA a	 */
      {	 { 0x3094, "lower"  }, { 0,1,0 }  },	   /* HIRAGANA u"	 */
      {	 { 0x30A1, "lower"  }, { 0,1,0 }  },	   /* KATAKANA a	 */
      {	 { 0x30FA, "lower"  }, { 0,1,0 }  },	   /* KATAKANA wo"	 */
      {	 { 0xFF66, "lower"  }, { 0,1,0 }  },	   /* HALF KATA WO	 */
      {	 { 0xFF6F, "lower"  }, { 0,1,0 }  },	   /* HALF KATA tu	 */
      {	 { 0xFF70, "lower"  }, { 0,1,0 }  },	   /* HALF KATA PL -	 */
      /* 124: */
      {	 { 0xFF71, "lower"  }, { 0,1,0 }  },	   /* HALF KATA A	 */
      {	 { 0xFF9E, "lower"  }, { 0,1,0 }  },	   /* HALF KATA MI	 */
      {	 { 0xFF71, "upper"  }, { 0,1,0 }  },	   /* HALF KATA A	 */
      {	 { 0xFF19, "xdigit" }, { 0,1,0 }  },	   /* FULL 9		 */
      {	 { 0x3000, "jspace" }, { 0,0,0 }  },	   /* IDEO. SPACE	 */
      /* Non jis? */
      {	 { 0x303F, "jspace" }, { 0,1,0 }  },	   /* IDEO.HALF SPACE	 */
      {	 { 0xFF19, "jdigit" }, { 0,0,0 }  },	   /* FULL 9		 */
      {	 { 0x3041, "jhira"  }, { 0,0,0 }  },	   /* HIRAGANA a	 */
      {	 { 0x3094, "jhira"  }, { 0,1,0 }  },	   /* HIRAGANA u"	 */
      {	 { 0x30A1, "jkata"  }, { 0,0,0 }  },	   /* KATAKANA a	 */
      /* Non jis: */
      {	 { 0x30FA, "jkata"  }, { 0,1,0 }  },	   /* KATAKANA wo"	 */
      {	 { 0xFF66, "jkata"  }, { 0,0,0 }  },	   /* HALF KATA WO	 */
      {	 { 0xFF6F, "jkata"  }, { 0,0,0 }  },	   /* HALF KATA tu	 */
      {	 { 0x4E05, "jkanji" }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      /* XXX This character does not exist in EUC-JP.  */
      {	 { 0x4E06, "jkanji" }, { 0,1,0 }  },	   /* CJK UNI.IDEO.NON-J */
      {	 { 0x4E07, "jkanji" }, { 0,0,0 }  },	   /* CJK UNI.IDEO.	 */
      { .is_last = 1 }
    }
  },
  {
    { Tiswctype, TST_LOC_end }
  }
};


/* dat_isw-funcs.c */
