/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wcsrtombs.c
 *
 *	 WCSTOMBS:  size_t wcsrtombs (char *s, const wchar_t **ws,
 *				      size_t n, mbstate *ps)
 */


/*
 *  CAUTION:
 *	     Do not use a value 0x01 for string data. The test program
 *	     uses it.
 *
 */


TST_WCSRTOMBS tst_wcsrtombs_loc [] = {
  {
    { Twcsrtombs, TST_LOC_de },
    {
      /* #01 : Any chars including a null char should not be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 0, 0, 0},
	/*expect*/ { 0,1,0,	 ""					},
      },
      /* #02 : Only one chars should be stored in s. No null termination.  */
      { /*input.*/ { 1,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 1, 0, 0 },
	/*expect*/ { 0,1,1,	 "Ä"					 },
      },
      /* #03 : Only two chars should be stored in s. No null termination.  */
      { /*input.*/ { 1,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 2, 0, 0 },
	/*expect*/ { 0,1,2,	 "ÄÖ"					 },
      },
      /* #04 : Only three chars should be stored in s. No null
	       termination.  */
      { /*input.*/ { 1,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 3, 0, 0 },
	/*expect*/ { 0,1,3,	 "ÄÖÜ"					 },
      },
      /* #05 : Only three chars should be stored in s with a null
	       termination. */
      { /*input.*/ { 1,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 4, 0, 0 },
	/*expect*/ { 0,1,3,	 "ÄÖÜ"					 },
      },
      /* #06 : Only three chars should be stored in s with a null
	       termination. */
      { /*input.*/ { 1,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 5, 0, 0 },
	/*expect*/ { 0,1,3,	 "ÄÖÜ"					 },
      },
      /* #07 : Invalid mb sequence. No chars should be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x0201,0x0221,0x0000,0x0000 }, 2, 0, 0 },
	/*expect*/ { EILSEQ,1,(size_t)-1,	 ""			 },
      },
      /* #08 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 0, 0, 0 },
	/*expect*/ { 0,1,3,	 ""					 },
      },
      /* #09 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 1, 0, 0 },
	/*expect*/ { 0,1,3,	 ""					 },
      },
      /* #10 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x00C4,0x00D6,0x00DC,0x0000 }, 5, 0, 0 },
	/*expect*/ { 0,1,3,	 ""					 },
      },
      /* #11 : s is a null pointer. No chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0201,0x0221,0x0000,0x0000 }, 5, 0, 0 },
	/*expect*/ { EILSEQ,1,(size_t)-1,	 ""			 },
      },
      /* #12 : ws is a null wc string, no chars should be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x0000 },			 5, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      /* #13 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0000 },			 5, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcsrtombs, TST_LOC_enUS },
    {
      /* #01 : Any chars including a null char should not be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 0, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      /* #02 : Only one chars should be stored in s. No null termination.  */
      { /*input.*/ { 1,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 1, 0, 0 },
	/*expect*/ { 0,1,1,	 "A"					 },
      },
      /* #03 : Only two chars should be stored in s. No null termination. */
      { /*input.*/ { 1,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 2, 0, 0 },
	/*expect*/ { 0,1,2,	 "AB"					 },
      },
      /* #04 : Only three chars should be stored in s. No null
	       termination.  */
      { /*input.*/ { 1,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 3, 0, 0 },
	/*expect*/ { 0,1,3,	 "ABC"					 },
      },
      /* #05 : Only three chars should be stored in s with a null
	       termination. */
      { /*input.*/ { 1,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 4, 0, 0 },
	/*expect*/ { 0,1,3,	 "ABC"					 },
      },
      /* #06 : Only three chars should be stored in s with a null
	       termination. */
      { /*input.*/ { 1,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 5, 0, 0 },
	/*expect*/ { 0,1,3,	 "ABC"					 },
      },
      /* #07 : Invalid mb sequence. No chars should be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x0201,0x0221,0x0000,0x0000 }, 2, 0, 0 },
	/*expect*/ { EILSEQ,1,(size_t)-1,	 ""					},
      },
      /* #08 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 0, 0, 0 },
	/*expect*/ { 0,1,3,	 ""					 },
      },
      /* #09 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 1, 0, 0 },
	/*expect*/ { 0,1,3,	 ""					 },
      },
      /* #10 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0041,0x0042,0x0043,0x0000 }, 5, 0, 0 },
	/*expect*/ { 0,1,3,	 ""					 },
      },
      /* #11 : s is a null pointer. No chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0201,0x0221,0x0000,0x0000 }, 5, 0, 0 },
	/*expect*/ { EILSEQ,1,(size_t)-1,	 ""			 },
      },
      /* #12 : ws is a null wc string, no chars should be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x0000 },			 5, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      /* #13 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0000 },			 5, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcsrtombs, TST_LOC_eucJP },
    {

      /* #01 : Any chars including a null char should not be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 0, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      /* #02 : Only one chars should be stored in s. No null termination.  */
      { /*input.*/ { 1,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 2, 0, 0 },
	/*expect*/ { 0,1,2,	    "\244\242"				 },
      },
      /* #03 : Only two chars should be stored in s. No null termination.  */
      { /*input.*/ { 1,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 4, 0, 0 },
	/*expect*/ { 0,1,4,	    "\244\242\244\244"			 },
      },
      /* #04 : Only three chars should be stored in s. No null
	       termination.  */
      { /*input.*/ { 1,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 6, 0, 0 },
	/*expect*/ { 0,1,6,	    "\244\242\244\244\216\263"		 },
      },
      /* #05 : Only three chars should be stored in s with a null
	       termination. */
      { /*input.*/ { 1,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 7, 0, 0 },
	/*expect*/ { 0,1,6,	    "\244\242\244\244\216\263"		 },
      },
      /* #06 : Only three chars should be stored in s with a null
	       termination. */
      { /*input.*/ { 1,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 8, 0, 0 },
	/*expect*/ { 0,1,6,	    "\244\242\244\244\216\263"		 },
      },
      /* #07 : Invalid mb sequence. No chars should be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x0201,0x0221,0x0000,0x0000 }, 2, 0, 0 },
	/*expect*/ { EILSEQ,1,-1, ""					 },
      },
      /* #08 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 0, 0, 0 },
	/*expect*/ { 0,1,6,	 ""					 },
      },
      /* #09 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 1, 0, 0 },
	/*expect*/ { 0,1,6,	 ""					 },
      },
      /* #10 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x3042,0x3044,0xFF73,0x0000 }, 8, 0, 0 },
	/*expect*/ { 0,1,6,	 ""					 },
      },
      /* #11 : s is a null pointer. No chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0201,0x0221,0x0000,0x0000 }, 5, 0, 0 },
	/*expect*/ { EILSEQ,1,(size_t)-1,	 ""			 },
      },
      /* #12 : ws is a null wc string, no chars should be stored in s.  */
      { /*input.*/ { 1,1,	{ 0x0000 },			 5, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      /* #13 : s is a null pointer, no chars should be stored in s.  */
      { /*input.*/ { 0,1,	{ 0x0000 },			 5, 0, 0 },
	/*expect*/ { 0,1,0,	 ""					 },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcsrtombs, TST_LOC_end }
  }
};
