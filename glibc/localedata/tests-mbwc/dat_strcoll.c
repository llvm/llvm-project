/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE: dat_strcoll.c
 *
 *	 STRCOLL:  int strcoll (const char *s1, const char *s2);
 */

/*
   NOTE:

   If a return value is expected to be 0, set ret_flg=1 and the
   expected value = 0.	If a return value is expected to be a
   positive/negative value, set ret_flg=0, and set the expected value
   = +1/-1.
   There is inconsistensy between tst_strcoll() and tst_wcscoll()(it
   has cmp_flg) for input data. I'll fix it.

   Assuming en_US to be en_US.ascii. (maybe, should be iso8859-1).



   ASCII CODE  : A,B,C, ...  , a, b, c, ...	 B,a:-1	  a,B:+1
   DICTIONARY : A,a,B,b,C,c,....  a,B:-1 B,a:+1 */

TST_STRCOLL tst_strcoll_loc [] = {
  {
    { Tstrcoll, TST_LOC_de },
    {
      { /*input.*/ { "ÄBCDEFG", "ÄBCDEFG"	      },  /* #1 */
	/*expect*/ { 0,1,0,			      },
      },
      { /*input.*/ { "XX Ä XX", "XX B XX"	      },  /* #2 */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "XX B XX", "XX Ä XX"	      },  /* #3 */
	/*expect*/ { 0,0,+1,			      },
      },
      { /*input.*/ { "B",	"a"		      },  /* #4 */
	/*expect*/ { 0,0,+1,			      },
      },
      { /*input.*/ { "a",	"B"		      },  /* #5 */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "b",	"A"		      },  /* #6 */
	/*expect*/ { 0,0,+1,			      },
      },
      { /*input.*/ { "A",	"b"		      },  /* #7 */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "ä",	"B"		      },  /* #8 */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "B",	"ä"		      },  /* #9 */
	/*expect*/ { 0,0,+1,			      },
      },
      { .is_last = 1 } /* Last element.  */
    }
  },
  {
    { Tstrcoll, TST_LOC_enUS },
    {
      { /*input.*/ { "ABCDEFG", "ABCDEFG"	      },  /* #1 */
	/*expect*/ { 0,1,0,			      },
      },
      { /*input.*/ { "XX a XX", "XX B XX"	      },  /* #2 */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "XX B XX", "XX a XX"	      },  /* #3 */
	/*expect*/ { 0,0,+1,			      },
      },
      {
	/* <WAIVER> */
	/*input.*/ { "B",	"a"		      },  /* #4 */
		   /* XXX We are not testing the C locale.  */
	/*expect*/ { 0,0,+1,			      },
      },
      {
	/* <WAIVER> */
	/*input.*/ { "a",	"B"		      },  /* #5 */
		   /* XXX We are not testing the C locale.  */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "b",	"A"		      },  /* #6 */
	/*expect*/ { 0,0,+1,			      },
      },
      { /*input.*/ { "A",	"b"		      },  /* #7 */
	/*expect*/ { 0,0,-1,			      },
      },
#ifdef NO_WAIVER
      /* XXX I do not yet know whether strcoll really should reject
	 characters outside the multibyte character range.  */
      {
	/* #8 */  /* <WAIVER> */
	/*input.*/ { "\244\242\244\244\244\246\244\250\244\252", "ABCDEFG" },
	/*expect*/ { EINVAL,0,0,		      },
      },
      {
	/* #9 */  /* <WAIVER> */
	/*input.*/ { "ABCZEFG", "\244\242\244\244\244\246\244\250\244\252" },
	/*expect*/ { EINVAL,0,0,		      },
      },
#endif
      { .is_last = 1 } /* Last element.  */
    }
  },
  {
    { Tstrcoll, TST_LOC_eucJP },
    {
      { /*input.*/ { "\244\242\244\244\244\246\244\250\244\252",
		     "\244\242\244\244\244\246\244\250\244\252" },  /* #1 */
	/*expect*/ { 0,1,0,			      },
      },
      { /*input.*/ { "\244\242\244\244\244\246\244\250\244\252",
		     "\244\242\244\244\244\363\244\250\244\252" },  /* #2 */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "\244\242\244\244\244\363\244\250\244\252",
		     "\244\242\244\244\244\246\244\250\244\252" },  /* #3 */
	/*expect*/ { 0,0,+1,			      },
      },
      { /*input.*/ { "B",	"a"		      },  /* #4 */
	/*expect*/ { 0,0,-1,			      },
      },
      { /*input.*/ { "a",	"B"		      },  /* #5 */
	/*expect*/ { 0,0,+1,			      },
      },
      { /*input.*/ { "b",	"A"		      },  /* #6 */
	/*expect*/ { 0,0,+1,			      },
      },
      { /*input.*/ { "A",	"b"		      },  /* #7 */
	/*expect*/ { 0,0,-1,			      },
      },
#ifdef NO_WAIVER
      /* XXX I do not yet know whether strcoll really should reject
	 characters outside the multibyte character range.  */
      {
	/* <WAIVER> */
	/*input.*/ { "\200\216\217", "ABCDEFG"	      },  /* #8 */
	/*expect*/ { EINVAL,0,0,		      },
      },
      {
	/* <WAIVER> */
	/*input.*/ { "ABCZEFG", "\200\216\217"	      },  /* #9 */
	/*expect*/ { EINVAL,0,0,		      },
      },
#endif
      { .is_last = 1 } /* Last element.  */
    }
  },
  {
    { Tstrcoll, TST_LOC_end }
  }
};
