/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wctomb.c
 *
 *	 WCTOMB:  int wctomb (char *s, wchar_t wc)
 */


/*
 *  FUNCTION:
 *
 *	  int  wctomb (char *s, wchar_t wc);
 *
 *	       return: the number of bytes
 *
 *  NOTE:
 *
 *	 o When you feed a null pointer for a string (s) to the function,
 *	   set s_flg=0 instead of putting just a 'NULL' there.
 *	   Even if you put a 'NULL', it means a null string as well as "".
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
 *	   state- dependency in this test function: tst_wctomb(), set
 *	   ret_flg=0 always. It's a special case, and the test
 *	   function takes care of it.
 *
 *	      Input	  Expect
 *
 *		s_flg=0		  ret_flg=0
 *		|		  |
 *	      { 0, 0 },	  { 0, 0, 0,  x,  "" }
 *		   |		      |
 *		   not used	      ret_val: 0/+1
 * (expected val)
 */


TST_WCTOMB tst_wctomb_loc [] = {
  {
    { Twctomb, TST_LOC_de },
    {
      /* #01 : normal case		   */
      { /*input.*/ { 1,	   0x00C4  },
	/*expect*/ { 0,1,1,  "Ä"	   },
      },
      /* #02 : normal case		   */
      { /*input.*/ { 1,	   0x00DC  },
	/*expect*/ { 0,1,1,  "Ü"	   },
      },
      /* #03 : normal case		   */
      { /*input.*/ { 1,	   0x0092  },
	/*expect*/ { 0,1,1,  "\222"  },
      },
      /* #04 : error case		   */
      { /*input.*/ { 1,	   0x3041  },
	/*expect*/ { 0,1,-1, ""	   },
      },
      /* #05 : state dependency	   */
      { /*input.*/ { 0,	   0x0000  },
	/*expect*/ { 0,0,0,  ""	   },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twctomb, TST_LOC_enUS },
    {
      /* #01 : normal case		   */
      { /*input.*/ { 1,	   0x0041  },
	/*expect*/ { 0,1,1,  "A"	   },
      },
      /* #02 : normal case		   */
      { /*input.*/ { 1,	   0x0042  },
	/*expect*/ { 0,1,1,  "B"	   },
      },
      /* #03 : error case		   */
      /* <WAIVER> */
      { /*input.*/ { 1,	   0x00C4  },
	/*expect*/ { 0,1,-1, ""	   },
      },
      /* #04 : error case		   */
      { /*input.*/ { 1,	   0x30A4  },
	/*expect*/ { 0,1,-1, ""	   },
      },
      /* #05 : state dependency	   */
      { /*input.*/ { 0,	   0x0000  },
	/*expect*/ { 0,0,0,  ""	   },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twctomb, TST_LOC_eucJP },
    {
      /* #01 : normal case		   */
      { /*input.*/ { 1,	   0x3042  },
	/*expect*/ { 0,1,2,  "\244\242"   },
      },
      /* #02 : normal case		   */
      { /*input.*/ { 1,	   0x3044  },
	/*expect*/ { 0,1,2,  "\244\244"   },
      },
      /* #03 : normal case		   */
      { /*input.*/ { 1,	   0x008E  },
	/*expect*/ { 0,1,-1, ""	   },
      },
      /* #04 : jisX0212		   */
      { /*input.*/ { 1,	   0x00C4	  },
	/*expect*/ { 0,1,3,  "\217\252\243" }, /* jisx0210  returns 3 */
      },
      /* #05 : state dependency	   */
      { /*input.*/ { 0,	   0x008E  },
	/*expect*/ { 0,0,0,  ""	   },
      },
      { .is_last = 1 }
    }
  },
  {
    { Twctomb, TST_LOC_end }
  }
};
