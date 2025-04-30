/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE: dat_wcscoll.c
 *
 *	 WCSCOLL:  int	wcscoll (const wchar_t *ws1, const wchar_t *ws2);
 */

/*
 *  CAUTION:
 *	     When LC_COLLATE (or LC_ALL) is set for ja_JP.EUC,
 *	     wcscoll() core-dumps for big values such as 0x3041
 *	     (0x0041 is okay) in glibc 2.1.2.
 *
 *  NOTE:
 *    a) When 0 is expected as a return value, set ret_flg=1.
 *	 - the return value is compared with an expected value: ret_val.
 *    b) When a positive value is expected as a return value,
 *	 set ret_flg=0 and set cmp_flg=+1.
 *	 - the return value is not compared with the expected value
 *	   (can not be compared); instead, the test program checks
 *	   if the return value is positive when cmp_flg=+1.
 *    c) When a negative value is expected as a return value,
 *	 ......
 *    d) When data contains invalid values, set err_val to the expected errno.
 *	 Set ret_flg=0 and cmp_flg=0 so that it doesn't compare
 *	 the return value with an expected value or doesn't check
 *	 the sign of the return value.
 *
 *
 *	     -------------------------------------------
 *	     CASE  err_val   ret_flg  ret_val	 cmp_flg
 *	     -------------------------------------------
 *	      a)      0	 1	  0	    0
 *	      b)      0	 0	  0	   +1
 *	      c)      0	 0	  0	   -1
 *	      d)    EINVAL	 0	  0	    0
 *	     -------------------------------------------
 */


TST_WCSCOLL tst_wcscoll_loc [] = {

    {	{ Twcscoll, TST_LOC_de },
	{
	  { /*input.*/ { { 0x00E1,0x00E2,0x00E3,0x0000 },
			 { 0x00E1,0x00E2,0x00E3,0x0000 }, },  /* #1 */
	    /*expect*/ { 0,1,0, 0,			  },
	  },
	  { /*input.*/ { { 0x0000,0x00E1,0x00E3,0x0000 },
			 { 0x0000,0x00E2,0x00E3,0x0000 }, },  /* #2 */
	    /*expect*/ { 0,1,0, 0,			  },
	  },
	  { /*input.*/ { { 0x00E1,0x00E1,0x00E3,0x0000 },
			 { 0x0000,0x00E2,0x00E3,0x0000 }, },  /* #3 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x0000,0x00E2,0x00E3,0x0000 },
			 { 0x00E1,0x00E1,0x00E3,0x0000 }, },  /* #4 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x00E1,0x0042,0x00E3,0x0000 },
			 { 0x00E1,0x0061,0x00E3,0x0000 }, },  /* #5 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x00E1,0x0061,0x00E3,0x0000 },
			 { 0x00E1,0x0042,0x00E3,0x0000 }, },  /* #6 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x00E1,0x00E2,0x0000	       },
			 { 0x00E1,0x00E2,0x00E9,0x0000 }, },  /* #7 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x00E1,0x00E2,0x00E9,0x0000 },
			 { 0x00E1,0x00E2,0x0000	       }, },  /* #8 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x00E1,0x0092,0x00E9,0x0000 },
			 { 0x00E1,0x008E,0x00E9,0x0000 }, },  /* #9 */
	    /*expect*/ { 0,0,0, +1,		       },
	  },
	  { /*input.*/ { { 0x00E1,0x008E,0x00E9,0x0000 },
			 { 0x00E1,0x0092,0x00E9,0x0000 }, },  /* #10 */
	    /*expect*/ { 0,0,0, -1,		       },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twcscoll, TST_LOC_enUS },
	{
	  { /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 },
			 { 0x0041,0x0042,0x0043,0x0000 }, },  /* #1 */
	    /*expect*/ { 0,1,0, 0,			  },
	  },
	  { /*input.*/ { { 0x0000,0x0041,0x0043,0x0000 },
			 { 0x0000,0x0042,0x0043,0x0000 }, },  /* #2 */
	    /*expect*/ { 0,1,0, 0,			  },
	  },
	  { /*input.*/ { { 0x0041,0x0041,0x0043,0x0000 },
			 { 0x0000,0x0042,0x0043,0x0000 }, },  /* #3 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x0000,0x0042,0x0043,0x0000 },
			 { 0x0041,0x0041,0x0043,0x0000 }, },  /* #4 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  /* XXX Correct order is lowercase before uppercase.  */
	  { /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 },
			 { 0x0041,0x0061,0x0043,0x0000 }, },  /* #5 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x0041,0x0061,0x0043,0x0000 },
			 { 0x0041,0x0042,0x0043,0x0000 }, },  /* #6 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x0041,0x0042,0x0000	       },
			 { 0x0041,0x0042,0x0049,0x0000 }, },  /* #7 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x0041,0x0042,0x0049,0x0000 },
			 { 0x0041,0x0042,0x0000	       }, },  /* #8 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  /* Do not assume position of character out of range.  */
	  { /*input.*/ { { 0x0041,0x0092,0x0049,0x0000 },
			 { 0x0041,0x008E,0x0049,0x0000 }, },  /* #9 */
	    /*expect*/ { 0,0,0, 0,		       },
	  },
	  { /*input.*/ { { 0x0041,0x008E,0x0049,0x0000 },
			 { 0x0041,0x0092,0x0049,0x0000 }, },  /* #10 */
	    /*expect*/ { 0,0,0, 0,		       },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twcscoll, TST_LOC_eucJP },
	{
	  { /*input.*/ { { 0x3041,0x3042,0x3043,0x0000 },
			 { 0x3041,0x3042,0x3043,0x0000 }, },  /* #1 */
	    /*expect*/ { 0,1,0, 0,			  },
	  },
	  { /*input.*/ { { 0x0000,0x3041,0x3043,0x0000 },
			 { 0x0000,0x3042,0x3043,0x0000 }, },  /* #2 */
	    /*expect*/ { 0,1,0, 0,			  },
	  },
	  { /*input.*/ { { 0x3041,0x3041,0x3043,0x0000 },
			 { 0x0000,0x3042,0x3043,0x0000 }, },  /* #3 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x0000,0x3042,0x3043,0x0000 },
			 { 0x3041,0x3041,0x3043,0x0000 }, },  /* #4 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x3041,0x0042,0x3043,0x0000 },
			 { 0x3041,0x0061,0x3043,0x0000 }, },  /* #5 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x3041,0x0061,0x3043,0x0000 },
			 { 0x3041,0x0042,0x3043,0x0000 }, },  /* #6 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x3041,0x3042,0xFF71,0x0000 },
			 { 0x3041,0x3042,0x30A2,0x0000 }, },  /* #7 */
	    /*expect*/ { 0,0,0, -1,			  },
	  },
	  { /*input.*/ { { 0x3041,0x3042,0x30A2,0x0000 },
			 { 0x3041,0x3042,0xFF71,0x0000 }, },  /* #8 */
	    /*expect*/ { 0,0,0, +1,			  },
	  },
	  { /*input.*/ { { 0x30FF,0x3092,0x3049,0x0000 },
			 { 0x3041,0x308E,0x3049,0x0000 }, },  /* #9 */
	    /*expect*/ { 0,0,0, -1,		       },
	  },
	  { /*input.*/ { { 0x3041,0x308E,0x3049,0x0000 },
			 { 0x30FF,0x3092,0x3049,0x0000 }, },  /* #10 */
	    /*expect*/ { 0,0,0, +1,		       },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twcscoll, TST_LOC_end } }
};
