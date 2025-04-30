/*
 *  TEST SUITE FOR MB/WC FUNCTIONS IN C LIBRARY
 *
 *	 FILE:	dat_wcschr.c
 *
 *	 WCSCHR:  wchar_t  *wcschr (const wchar_t *ws, wchar_t wc);
 */

TST_WCSCHR tst_wcschr_loc [] = {

    {	{ Twcschr, TST_LOC_de },
	{
	  { /*input.*/ { { 0x00C1,0x00C2,0x00C3,0x0000 }, 0x00C0 },  /* #1 */
	    /*expect*/ { 0,1,(wchar_t *)NULL },
	  },
	  { /*input.*/ { { 0x00C1,0x00C2,0x00C3,0x0000 }, 0x00C1 },  /* #2 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x00C1,0x00C2,0x00C3,0x0000 }, 0x00C2 },  /* #3 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x00C1,0x00C2,0x00C3,0x0000 }, 0x00C3 },  /* #4 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x00C1,0x00C2,0x00C3,0x0000 }, 0x0000 },  /* #5 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x0000,0x00C2,0x00C3,0x0000 }, 0x00C1 },  /* #6 */
	    /*expect*/ { 0,1,(wchar_t *)NULL },
	  },
	  { /*input.*/ { { 0x0000,0x00C2,0x00C3,0x0000 }, 0x0000 },  /* #7 */
	    /*expect*/ { 0,0,0 },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twcschr, TST_LOC_enUS },
	{
	  { /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 }, 0x0040 },  /* #1 */
	    /*expect*/ { 0,1,(wchar_t *)NULL },
	  },
	  { /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 }, 0x0041 },  /* #2 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 }, 0x0042 },  /* #3 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 }, 0x0043 },  /* #4 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x0041,0x0042,0x0043,0x0000 }, 0x0000 },  /* #5 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x0000,0x0042,0x0043,0x0000 }, 0x0041 },  /* #6 */
	    /*expect*/ { 0,1,(wchar_t *)NULL },
	  },
	  { /*input.*/ { { 0x0000,0x0042,0x0043,0x0000 }, 0x0000 },  /* #7 */
	    /*expect*/ { 0,0,0 },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twcschr, TST_LOC_eucJP },
	{
	  { /*input.*/ { { 0x3041,0x3042,0x3043,0x0000 }, 0x3040 },  /* #1 */
	    /*expect*/ { 0,1,(wchar_t *)NULL },
	  },
	  { /*input.*/ { { 0x3041,0x3042,0x3043,0x0000 }, 0x3041 },  /* #2 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x3041,0x3042,0x3043,0x0000 }, 0x3042 },  /* #3 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x3041,0x3042,0x3043,0x0000 }, 0x3043 },  /* #4 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x3041,0x3042,0x3043,0x0000 }, 0x0000 },  /* #5 */
	    /*expect*/ { 0,0,0 },
	  },
	  { /*input.*/ { { 0x0000,0x3042,0x3043,0x0000 }, 0x3041 },  /* #6 */
	    /*expect*/ { 0,1,(wchar_t *)NULL },
	  },
	  { /*input.*/ { { 0x0000,0x3042,0x3043,0x0000 }, 0x0000 },  /* #7 */
	    /*expect*/ { 0,0,0 },
	  },
	  { .is_last = 1 }
	}
    },
    {	{ Twcschr, TST_LOC_end } }
};
