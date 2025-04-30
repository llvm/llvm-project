/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wcrtomb.c
 *
 *	 WCRTOMB:  intwcrtomb (char *s, wchar_t wc, mbstate_t *ps);
 *
 */

TST_WCRTOMB tst_wcrtomb_loc [] = {
  {
    { Twcrtomb, TST_LOC_de },
    {
      /* #01 : normal case			       */
      { /*input.*/ { 1,		 0x00FC,   0,0 },
	/*expect*/ { 0,	   1,1,	 "ü"	       },
      },
      /* #02 : normal case			       */
      { /*input.*/ { 1,		 0x00D6,   0,0 },
	/*expect*/ { 0,	   1,1,	 "Ö"	       },
      },
      /* #03 : error case			       */
      { /*input.*/ { 1,		 0xFFA1,   0,0 },
	/*expect*/ {  EILSEQ,1,-1, ""	       },
      },
      /* #04 :				       */
      { /*input.*/ { 0,		 0x0041,   0,0 },
	/*expect*/ { 0,	   1,1,	 ""	       },
      },
      /* #05 :				       */
      { /*input.*/ { 0,		 0x0092,   0,0 },
	/*expect*/ { 0,	   1,1,	 ""	       },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcrtomb, TST_LOC_enUS },
    {
      /* #01 : normal case			       */
      { /*input.*/ { 1,		 0x0041,   0,0 },
	/*expect*/ { 0,	   1,1,	 "A"	       },
      },
      /* #02 : normal case			       */
      { /*input.*/ { 1,		 0x0042,   0,0 },
	/*expect*/ { 0,	   1,1,	 "B"	       },
      },
      /* #03 : error case			       */
      /* <WAIVER> x 2 */
      { /*input.*/ { 1,		 0x0092,   0,0 },  /* assume ascii */
	/*expect*/ {  EILSEQ,1,-1, ""	       },
      },
      /* #04 :				       */
      { /*input.*/ { 0,		 0x0041,   0,0 },
	/*expect*/ { 0,	   1,1,	 ""	       },
      },
      /* #05 :				       */
      { /*input.*/ { 0,		 0x0092,   0,0 },
	/*expect*/ { 0,	   1,1,	 ""	       },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcrtomb, TST_LOC_eucJP },
    {
      /* #01 : normal case			       */
      { /*input.*/ { 1,		 0x3042,   0,0 },
	/*expect*/ { 0,      1,2,  "\244\242"	   },
      },
      /* #02 : normal case			       */
      { /*input.*/ { 1,		 0x3044,   0,0 },
	/*expect*/ { 0,      1,2,  "\244\244"	   },
      },
      /* #03 : normal case			       */
      { /*input.*/ { 1,		 0x008E,   0,0 },
	/*expect*/ { EILSEQ, 1,-1, ""	       },
      },
      /* #04 :				       */
      { /*input.*/ { 0,		 0x3042,   0,0 },
	/*expect*/ { 0,	   0,0,	 ""	       },
      },
      /* #05 :				       */
      { /*input.*/ { 0,		 0x008E,   0,0 },
	/*expect*/ { 0,	   0,0,	 ""	       },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcrtomb, TST_LOC_end }
  }
};
