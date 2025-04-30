/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_mblen.c
 *
 *	 MBLEN:	 int mblen (char *s, size_t n);
 */


/*
 *  NOTE:
 *	  int  mblen (char *s, size_t n);
 *
 *	  where	     n: a maximum number of bytes
 *
 *	  return - the number of bytes
 *
 *  CAUTION:
 *
 *	 o When you feed a null pointer for a string (s) to the function,
 *	   set s_flg=0 instead of putting just a 'NULL' there.
 *	   Even if you set a 'NULL', it doens't mean a NULL pointer.
 *
 *	 o When s is a null pointer, the function checks state dependency.
 *
 *	       state-dependent encoding	     - return  NON-zero
 *	       state-independent encoding    - return  0
 *
 *	   If state-dependent encoding is expected, set
 *
 *	       s_flg = 0,  ret_flg = 0,	 ret_val = +1
 *
 *	   If state-independent encoding is expected, set
 *
 *	       s_flg = 0,  ret_flg = 0,	 ret_val = 0
 *
 *
 *	   When you set ret_flg=1, the test program simply compares an
 *	   actual return value with an expected value. You can check
 *	   state-independent case (return value is 0) in that way, but
 *	   you can not check state-dependent case. So when you check
 *	   state- dependency in this test function: tst_mblen(), set
 *	   ret_flg=0 always. It's a special case, and the test
 *	   function takes care of it.
 *
 *	       s_flg=0		 ret_flg=0
 *	       |		 |
 *	     { 0, 0 },	 { 0, 0, 0,  x }
 *		  |		     |
 *		  not used	     ret_val: 0/+1
 *				     (expected val) */


TST_MBLEN tst_mblen_loc [] = {
  {
    { Tmblen, TST_LOC_de },
    {
      /* 01: a character.  */
      {	 { 1, "\300",	   USE_MBCURMAX }, { 0,	1,  1 }	 },
      /* 02: a character.  */
      {	 { 1, "\309",	   USE_MBCURMAX }, { 0,	1,  1 }	 },
      /* 03: a character + an invalid byte.  */
      {	 { 1, "Z\204",	   USE_MBCURMAX }, { 0,	1, +1 }	 },
      /* 04: control/invalid characters.  */
      {	 { 1, "\177\000",  USE_MBCURMAX }, { 0,	1, +1 }	 },
      /* 05: a null string.  */
      {	 { 1, "",	   USE_MBCURMAX }, { 0,	1,  0 }	 },
      /* 06: a null pointer.  */
      {	 { 0, "",	   USE_MBCURMAX }, { 0,	0,  0 }	 },
      /* Last element.	*/
      {	 .is_last = 1 }
    }
  },
  {
    { Tmblen, TST_LOC_enUS },
    {
      /* 01: a character.  */
      {	 { 1, "A",	   USE_MBCURMAX }, { 0,	1,  1 }	 },
      /* 02: a character.  */
      {	 { 1, "a",	   USE_MBCURMAX }, { 0,	1,  1 }	 },
      /* 03: a character + an invalid byte.  */
      {	 { 1, "Z\204",	   USE_MBCURMAX }, { 0,	1, +1 }	 },
      /* 04: control/invalid characters.  */
      {	 { 1, "\177\000",  USE_MBCURMAX }, { 0,	1, +1 }	 },
      /* 05: a null string.  */
      {	 { 1, "",	   USE_MBCURMAX }, { 0,	1,  0 }	 },
      /* 06: a null pointer.  */
      {	 { 0, "",	   USE_MBCURMAX }, { 0,	0,  0 }	 },
      /* Last element.	*/
      {	 .is_last = 1 }
    }
  },
  {
    { Tmblen, TST_LOC_eucJP },
    {
      /* 01: a character.  */
      {	 { 1, "\264\301",	   USE_MBCURMAX }, { 0, 1,  2 }	 },
      /* 02: a character.  */
      {	 { 1, "\216\261",	   USE_MBCURMAX }, { 0, 1,  2 }  },
      /* 03: a character + an invalid byte.  */
      {	 { 1, "\260\241\200",	   USE_MBCURMAX }, { 0, 1,  2 }	 },
      /* 04: control/invalid characters.  */
      {	 { 1, "\377\202",  USE_MBCURMAX }, { EILSEQ, 1, -1 }	 },
      /* 05: a null string.  */
      {	 { 1, "",	   USE_MBCURMAX }, { 0,	1,  0 }	 },
      /* 06: a null pointer.  */
      {	 { 0, "",	   USE_MBCURMAX }, { 0,	0,  0 }	 },
      /* Last element.	*/
      {	 .is_last = 1 }
    }
  },
  {
    { Tmblen, TST_LOC_end}
  }
};
