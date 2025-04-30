/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN CLIBRARY
 *
 *	 FILE:	dat_strxfrm.c
 *
 *	 STRXFRM:  size_t strxfrm (char *s1, const char s2, size_t n);
 */


/*
 *  NOTE:
 *
 *  Return value and errno value are checked only for 2nd string:
 *  org2[]; n1 and n2 don't mean bytes to be translated.
 *  It means a buffer size including a null character.
 *  Results of this test depens on results of strcoll().
 *  If you got errors, check both test results.
 *
 *  The buffer size should be enough to contain a string including a
 *  null char.	Returns the number of bytes of the string (NOT
 *  including a null char).
 */



TST_STRXFRM tst_strxfrm_loc [] = {
  {
    { Tstrxfrm, TST_LOC_de },
    {
      { /*inp*/ { "\xf6\xc4\xe4\xfc", "\xf6\xc4\xe4\xfc", 17, 17 },  /* #01 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "aA", "Aa",	    10, 10 },  /* #02 */
	/*exp*/ { 0,0,0 ,			   },
      },
      { /*inp*/ { "Aa", "aA",	    10, 10 },  /* #03 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "abc", "",	    13, 13 },  /* #04 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "a", "B",		     7,	 7 },  /* #05 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "B", "a",		     7,	 7 },  /* #06 */
	/*exp*/ { 0,0,0,			   },
      },
      {
	/* hiragana == latin1 */
	/*inp*/ { "abc", "\244\241\244\242",  13,  9 },	 /* #07 */
	/*exp*/ { 0,0,0,		       },
      },
      { .is_last = 1 }
    }
  },
  {
    { Tstrxfrm, TST_LOC_enUS },
    {
      { /*inp*/ { "abcd", "abcd",	    17, 17 },  /* #01 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "aA", "Aa",	    10, 10 },  /* #02 */
	/*exp*/ { 0,0,0 ,			   },
      },
      { /*inp*/ { "Aa", "aA",	    10, 10 },  /* #03 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "abc", "",	    13, 13 },  /* #04 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "a", "B",		     7,	 7 },  /* #05 */
	/*exp*/ { 0,0,0,			   },
      },
      { /*inp*/ { "B", "a",		     7,	 7 },  /* #06 */
	/*exp*/ { 0,0,0,			   },
      },
#ifdef NO_WAIVER
      {
	/* <WAIVER> */
	/*inp*/ { "abc", "\244\241\244\242",  13,  9 },	 /* #07 */
	/*exp*/ { EINVAL,0,0,		       },
      },
#endif
      { .is_last = 1 }
    }
  },
  {
    { Tstrxfrm, TST_LOC_eucJP },	 /* ??? */
    {
      {
	/* #01 */
	/*inp*/ { "\244\242\244\241",  "\244\241\244\242",   5,	 5 },
	/*exp*/ { 0,0,0,		       },
      },
      {
	/* #02 */
	/*inp*/ { "\244\241\244\242",  "\244\242\244\241",   5,	 5 },
	/*exp*/ { 0,0,0,		       },
      },
      {
	/* #03 */
	/*inp*/ { "\244\242\216\261",  "\216\261\244\242",   5,	 5 },
	/*exp*/ { 0,0,0,		       },
      },
#ifdef NO_WAIVER
      {
	/*inp*/ { "AAA", "\216\217",	 5,  5 },  /* #04 */ /* <WAIVER> */
	/*exp*/ { EINVAL,0,0,		       },
      },
#endif
      { .is_last = 1 }
    }
  },
  {
    { Tstrxfrm, TST_LOC_end }
  }
};
