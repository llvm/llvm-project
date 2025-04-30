/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN CLIBRARY
 *
 *	 FILE:	dat_wcsxfrm.c
 *
 *	 WCSXFRM:  size_t  wcsxfrm (char *s1, const char s2, size_t n);
 */

/*
 *  NOTE:
 *
 *  Return value and errno value are checked only for 2nd string:
 *  org2[]; n1 and n2 don't mean bytes to be translated.
 *  It means a buffer size including a null character.
 *  Results of this test depens on results of wcscoll().
 *  If you got errors, check both test results.
 */


TST_WCSXFRM tst_wcsxfrm_loc [] = {

  {
    { Twcsxfrm, TST_LOC_de },
    {
      { /*inp*/ { { 0x00C1,0x0000 }, { 0x00C1,0x0000 }, 7, 7 },	 /* #01 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x0042,0x0000 }, { 0x0061,0x0000 }, 7, 7 },	 /* #02 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x0061,0x0000 }, { 0x0042,0x0000 }, 7, 7 },	 /* #03 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x00E4,0x0000 }, { 0x00DC,0x0000 }, 7, 7 },	 /* #04 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x00DC,0x0000 }, { 0x00E4,0x0000 }, 7, 7 },	 /* #05 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twcsxfrm, TST_LOC_enUS },
    {
      { /*inp*/ { { 0x0041,0x0000 }, { 0x0041,0x0000 }, 7, 7 },	 /* #01 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x0042,0x0000 }, { 0x0061,0x0000 }, 7, 7 },	 /* #02 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x0061,0x0000 }, { 0x0042,0x0000 }, 7, 7 },	 /* #03 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x0000,0x0000 }, { 0x0000,0x0000 }, 7, 7 },	 /* #04 */
	/*exp*/ {   0,	       0,0,		     },
      },
#ifdef NO_WAIVER
      { /* <WAIVER> x 2 */
	/*inp*/ { { 0x3061,0x0000 }, { 0xFF42,0x0000 }, 7, 7 },	 /* #05 */
	/* <WAIVER>	*/
	/*exp*/ {   EINVAL,	       1,(size_t)-1,	     },
      },
#endif
      { .is_last = 1 }
    }
  },
  {
    { Twcsxfrm, TST_LOC_eucJP },	     /* need more test data ! */
    {
      { /*inp*/ { { 0x3041,0x0000 }, { 0x3041,0x0000 }, 7, 7 },	 /* #01 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x0042,0x0000 }, { 0x0061,0x0000 }, 7, 7 },	 /* #02 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x0061,0x0000 }, { 0x0042,0x0000 }, 7, 7 },	 /* #03 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0x30A2,0x0000 }, { 0xFF71,0x0000 }, 7, 7 },	 /* #04 */
	/*exp*/ {   0,	       0,0,		     },
      },
      { /*inp*/ { { 0xFF71,0x0000 }, { 0x30A2,0x0000 }, 7, 7 },	 /* #05 */
	/*exp*/ {   0,	       0,0,		     },
      },
#ifdef NO_WAIVER
      /* <WAIVER> x 2 */
      { /*inp*/ { { 0x008E,0x0000 }, { 0x008F,0x0000 }, 7, 7 },	 /* #06 */
	/*exp*/ {   EINVAL,	       1,(size_t)-1,	     },
      },
#endif
      { .is_last = 1 }
    }
  },
  {
    { Twcsxfrm, TST_LOC_end }
  }
};
